from typing import Dict
import importlib
import copy

import torch as tc

from drl.envs.wrappers import Wrapper


def torch_dtype(np_dtype):
    module = importlib.import_module('torch')
    dtype = getattr(module, str(np_dtype))
    return dtype


class MetadataManager:
    def __init__(self, present_defaults):
        """
        Maintains running statistics on a stream of experience.

        Args:
            present_defaults: A dictionary of fields and default values.
        """
        self._fields = present_defaults.keys()
        self._present_defaults = present_defaults
        self._present_meta = copy.deepcopy(present_defaults)
        self._past_meta = {key: list() for key in self._fields}

    @property
    def present_meta(self):
        """
        A dictionary of tracked statistics, keyed by field name.
        """
        return self._present_meta

    @property
    def past_meta(self):
        """
        A dictionary of lists of tracked statistics, keyed by field name.
        Each stat in a field's list was computed by aggregating
            over a field-dependent timespan, such as a life or an episode.
        """
        return self._past_meta

    def update_present(self, deltas):
        """
        Update present_meta by incrementing each field by delta[field].
        """
        for key in deltas:
            self._present_meta[key] += deltas[key]

    def present_done(self, fields):
        """
        Update past_meta for listed fields by appending present_meta[field].
        """
        for key in fields:
            self._past_meta[key].append(self._present_meta[key])
            self._present_meta[key] = copy.deepcopy(self._present_defaults[key])

    def past_done(self):
        """
        Update past_meta by resetting to empty. To ensure correct aggregation
        of statistics is not interrupted by trajectory segment boundaries,
        present_meta is kept intact.
        """
        self._past_meta = {key: list() for key in self._fields}


class Trajectory:
    def __init__(self, obs_space, ac_space, rew_keys, seg_len, extra_steps):
        """
        Args:
            obs_space: Observation space.
            ac_space: Action space.
            rew_keys: Reward keys.
            seg_len: Segment length.
            extra_steps: Extra steps for n-step reward based credit assignment.
        """
        self._obs_space = obs_space
        self._ac_space = ac_space
        self._rew_keys = rew_keys
        self._seg_len = seg_len
        self._extra_steps = extra_steps
        self._timesteps = seg_len + extra_steps
        self._observations = None
        self._actions = None
        self._rewards = None
        self._dones = None
        self._erase()

    def _erase(self):
        obs_shape, ac_shape = self._obs_space.shape, self._ac_space.shape
        ac_dtype = torch_dtype(self._ac_space.dtype)
        self._observations = tc.zeros((self._timesteps+1, *obs_shape), dtype=tc.float32)
        self._actions = tc.zeros((self._timesteps+1, *ac_shape), dtype=ac_dtype)
        self._rewards = {
            k: tc.zeros(self._timesteps, dtype=tc.float32) for k in self._rew_keys
        }
        self._dones = tc.zeros(self._timesteps, dtype=tc.float32)

    def record(self, t, o_t, a_t, r_t, d_t):
        """
        Args:
            t: Integer index for the step to be stored.
            o_t: Observation.
            a_t: Action.
            r_t: Reward dict.
            d_t: Done signal.
        Returns:
            None.
        """
        i = t % self._timesteps
        self._observations[i] = tc.tensor(o_t).float()
        self._actions[i] = tc.tensor(a_t).long()
        for key in self._rew_keys:
            self._rewards[key][i] = tc.tensor(r_t[key]).float()
        self._dones[i] = tc.tensor(d_t).float()

    def record_nexts(self, o_Tp1, a_Tp1):
        self._observations[-1] = tc.tensor(o_Tp1).float()
        self._actions[-1] = tc.tensor(a_Tp1).long()

    def report(self):
        """
        Generates a dictionary of observations, actions, rewards, and dones.
        Erases the internally-stored trajectory.
        If self._extra_steps > 0, writes the last self._extra_steps timesteps
            of the old trajectory to the front of the the blank trajectory.

        Returns:
            Dict[str, tc.Tensor]: Trajectory data.
        """
        results = {
            'observations': self._observations,
            'actions': self._actions,
            'rewards': self._rewards,
            'dones': self._dones
        }
        self._erase()
        if self._extra_steps > 0:
            src = slice(self._seg_len, self._seg_len + self._extra_steps)
            dest = slice(0, self._extra_steps)
            self._observations[dest] = results['observations'][src]
            self._actions[dest] = results['actions'][src]
            for k in self._rew_keys:
                self._rewards[k][dest] = results['rewards'][k][src]
            self._dones[dest] = results['dones'][src]
        return results


class TrajectoryManager:
    def __init__(self, env, policy_net, seg_len, extra_steps):
        """
        Args:
            env: OpenAI gym environment or Wrapper instance.
            policy_net: Network with a 'policy' predictor attached.
            seg_len: Segment length.
            extra_steps: Extra steps for n-step reward based credit assignment.
        """
        self._env = env
        self._policy_net = policy_net
        self._seg_len = seg_len
        self._extra_steps = extra_steps

        self._o_t = self._env.reset()
        self._a_t = self._choose_action(self._o_t)
        self._trajectory = Trajectory(
            obs_shape=self._o_t.shape,
            ac_shape=self._a_t.shape,
            rew_keys=self._get_reward_keys(),
            seg_len=self._seg_len,
            extra_steps=self._extra_steps)
        self._metadata_mgr = MetadataManager(
            present_defaults={
                'ep_len': 0,
                'ep_ret': 0.,
                'ep_len_raw': 0,
                'ep_ret_raw': 0.
            }
        )
        _ = self.generate(initial=True)

    def _get_reward_keys(self):
        def spec_exists():
            if isinstance(self._env, Wrapper):
                return self._env.reward_spec is not None
            return False
        if spec_exists():
            return self._env.reward_spec.keys
        return {'extrinsic_raw', 'extrinsic'}

    def _choose_action(self, o_t):
        # todo(lucaslingle): add support for stateful policies
        # todo(lucaslingle): add support for RL^2/NGU-style inputting
        #    of past rewards as inputs
        predictions = self._policy_net(
            tc.tensor(o_t).float().unsqueeze(0), predict=['policy'], step=True)
        pi_dist_t = predictions.get('policy')
        a_t = pi_dist_t.sample().squeeze(0).detach().numpy()
        return a_t

    def _step_env(self, a_t):
        o_tp1, r_t, done_t, info_t = self._env.step(a_t)
        if not isinstance(r_t, dict):
            r_t = {'extrinsic_raw': r_t, 'extrinsic': r_t}
        return o_tp1, r_t, done_t, info_t

    @tc.no_grad()
    def generate(self, initial=False):
        """
        Generates a trajectory by stepping the environment.

        Args:
            initial: If true, steps the environment for self._extra_steps
               and saves the results without returning anything. Otherwise,
               the full trajectory is returned.

        Returns:
            Optional[Dict[str, tc.Tensor]]: Maybe some trajectory data.
        """
        # determine the time indices for the trajectory segment to generate
        if initial:
            if self._extra_steps == 0:
                return
            start_t, end_t = 0, self._extra_steps
        else:
            start_t, end_t = self._extra_steps, self._seg_len+self._extra_steps

        # generate a trajectory segment.
        self._policy_net.eval()
        for t in range(start_t, end_t):
            o_tp1, r_t, done_t, info_t = self._step_env(self._a_t)
            self._trajectory.record(t, self._o_t, self._a_t, r_t, done_t)
            self._metadata_mgr.update_present(
                deltas={
                    'ep_len': 1,
                    'ep_ret': r_t['extrinsic'],
                    'ep_len_raw': 1,
                    'ep_ret_raw': r_t['extrinsic_raw']
                }
            )
            if done_t:
                self._metadata_mgr.present_done(fields=['ep_len', 'ep_ret'])
                def was_real_done():
                    if 'ale.lives' in info_t:
                        return info_t['ale.lives'] == 0
                    return True
                if was_real_done():
                    self._metadata_mgr.present_done(
                        fields=['ep_len_raw', 'ep_ret_raw'])
                o_tp1 = self._env.reset()
            a_tp1 = self._choose_action(self._o_t)
            self._o_t = o_tp1
            self._a_t = a_tp1

        # if we've gotten here and initial is True,
        # then dont mess it up by generating a trajectory report, since it will
        # erase the extra_steps trajectory steps we just pre-generated.
        if initial:
            return

        # return results with next timestep observation and action included.
        # o_Tp1 is needed for value-based credit assignment.
        # a_Tp1 is included to simplify the implementation elsewhere.
        self._trajectory.record_nexts(o_Tp1=o_tp1, a_Tp1=a_tp1)
        results = {
            **self._trajectory.report(),
            'metadata': self._metadata_mgr.past_meta
        }
        self._metadata_mgr.past_done()
        return results
