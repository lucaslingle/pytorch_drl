from collections import Counter
import copy

import torch as tc

from drl.envs.wrappers import Wrapper


def global_mean(metric, world_size):
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.item() / world_size


def global_means(metrics, world_size):
    return Counter({k: global_mean(v, world_size) for k, v in metrics.items()})


class MetadataManager:
    def __init__(self, present_meta):
        self._fields = present_meta.keys()
        self._present_defaults = copy.deepcopy(present_meta)
        self._present_meta = present_meta
        self._past_meta = {key: list() for key in self._fields}

    def update_present(self, deltas):
        for key in deltas:
            self._present_meta[key] += deltas[key]

    def present_done(self, fields):
        for key in fields:
            self._past_meta[key].append(self._present_meta[key])
            self._present_meta[key] = copy.deepcopy(self._present_defaults[key])

    def past_done(self):
        self._past_meta = {key: list() for key in self._fields}

    @property
    def present_meta(self):
        return self._present_meta

    @property
    def past_meta(self):
        return self._past_meta


class Trajectory:
    def __init__(self, obs_shape, rew_keys, seg_len, extra_steps):
        self._obs_shape = obs_shape
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
        self._observations = tc.zeros((self._timesteps+1, *self._obs_shape), dtype=tc.float32)
        self._actions = tc.zeros(self._timesteps+1, dtype=tc.float32)
        self._rewards = {
            k: tc.zeros(self._timesteps, dtype=tc.float32) for k in self._rew_keys
        }
        self._dones = tc.zeros(self._timesteps, dtype=tc.float32)

    def record(self, t, o_t, a_t, r_t, d_t):
        i = t % self._timesteps
        self._observations[i] = tc.tensor(o_t).float()
        self._actions[i] = tc.tensor(a_t).long()
        for key in self._rew_keys:
            self._rewards[key][i] = tc.tensor(r_t[key]).float()
        self._dones[i] = tc.tensor(d_t).float()

    def report(self):
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
            self._actions[dest] = results['observations'][src]
            for k in self._rew_keys:
                self._rewards[k][dest] = results['rewards'][k][src]
            self._dones[dest] = results['dones'][src]
        return results


class TrajectoryManager:
    def __init__(self, env, policy_net, seg_len, extra_steps):
        self._env = env
        self._policy_net = policy_net
        self._seg_len = seg_len
        self._extra_steps = extra_steps

        self._o_t = self._env.reset()
        self._a_t = self._choose_action(self._o_t)
        self._trajectory = Trajectory(
            obs_shape=self._o_t.shape,
            rew_keys=self._get_reward_keys(),
            seg_len=self._seg_len,
            extra_steps=self._extra_steps)
        self._metadata_mgr = MetadataManager(
            present_meta={
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
        predictions = self._policy_net(
            tc.tensor(o_t).float().unsqueeze(0), predictions=['policy'])
        pi_dist_t = predictions.get('policy')
        a_t = pi_dist_t.sample().squeeze(0).detach().numpy()
        return a_t

    def _step_env(self, a_t):
        # step environment
        o_tp1, r_t, done_t, info_t = self._env.step(a_t)
        if not isinstance(r_t, dict):
            r_t = {'extrinsic_raw': r_t, 'extrinsic': r_t}
        return o_tp1, r_t, done_t, info_t

    @tc.no_grad()
    def generate(self, initial=False):
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
            # step environment
            o_tp1, r_t, done_t, info_t = self._step_env(self._a_t)

            # record everything
            self._trajectory.record(t, self._o_t, self._a_t, r_t, done_t)
            self._metadata_mgr.update_present(
                deltas={
                    'ep_len': 1,
                    'ep_ret': r_t['extrinsic'],
                    'ep_len_raw': 1,
                    'ep_ret_raw': r_t['extrinsic_raw']
                }
            )

            # handle metadata for episode boundaries
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

            # choose next action
            a_tp1 = self._choose_action(o_tp1)

            # save next observation and action
            self._o_t = o_tp1
            self._a_t = a_tp1

        # if we've gotten here and initial is True,
        # then dont mess it up by generating a trajectory report, since it will
        # erase the extra_steps trajectory steps we just prepended.
        if initial:
            return

        # return results with next timestep observation and action included.
        # o_Tp1 is needed for value-based credit assignment, and a_Tp1 is needed
        # for credit assignment in q-learning.
        results = {
            **self._trajectory.report(),
            'metadata': self._metadata_mgr.past_meta
        }
        results['observations'][-1] = o_tp1
        results['actions'][-1] = a_tp1
        self._metadata_mgr.past_done()
        return results
