from typing import Any, List, Union, Optional, Dict, Iterable
import importlib
import copy

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import gym

from drl.envs.wrappers import Wrapper
from drl.utils.typing import Action, Observation, DictReward, EnvOutput


def torch_dtype(dtype: str) -> tc.dtype:
    """
    Converts a string to a torch dtype.

    Args:
        dtype (str): A dtype string like uint8, int32, float64, etc.

    Returns:
        torch.dtype: The corresponding torch dtype.
    """
    module = importlib.import_module('torch')
    dtype = getattr(module, str(dtype))
    return dtype


class MetadataManager:
    def __init__(self, defaults: Dict[str, Any]):
        """
        Maintains running statistics on a stream of experience.

        Args:
            defaults (Dict[str, Any]): A dictionary mapping from names
                to default values. The each value must implement __add__.
        """
        self._keys = defaults.keys()
        self._defaults = defaults
        self._present = copy.deepcopy(defaults)
        self._pasts = {key: list() for key in self._keys}

    @property
    def keys(self):
        return self._keys

    @property
    def present(self) -> Dict[str, Any]:
        """
        A dictionary mapping from names to aggregated present values.
        """
        return self._present

    @property
    def pasts(self) -> Dict[str, List[Any]]:
        """
        A dictionary mapping from names to a list of aggregated past values.
        """
        return self._pasts

    def update_present(self, deltas: Dict[str, Any]) -> None:
        """
        Updates `self.present` by incrementing `present[key]` by `deltas[key]`.

        Args:
            deltas (Dict[str, Any]): A dictionary mapping a subset of
                `self.keys` to deltas to increment aggregated present values by.

        Returns:
            None.
        """
        assert set(deltas.keys()) <= set(self.keys)
        for key in deltas:
            self._present[key] += deltas[key]

    def present_done(self, *keys: Iterable[str]) -> None:
        """
        Updates `pasts` for listed fields by appending present[field].

        Args:
            *keys (Iterable[str]): A variable-length listing of keys whose present
                is done.

        Returns:
            None.
        """
        for key in keys:
            self._pasts[key].append(self._present[key])
            self._present[key] = copy.deepcopy(self._defaults[key])

    def past_done(self) -> None:
        """
        Resets `self.pasts` to a dictionary of empty lists.

        To ensure correct aggregation of statistics is not interrupted by
        trajectory segment boundaries, `self.present` is kept intact.

        Returns:
            None.
        """
        self._pasts = {key: list() for key in self._keys}


class Trajectory:
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            ac_space: gym.spaces.Space,
            rew_keys: List[str],
            seg_len: int,
            extra_steps: int):
        """
        Args:
            obs_space (gym.spaces.Space): Observation space.
            ac_space (gym.spaces.Space): Action space.
            rew_keys (List[str]): Reward keys.
            seg_len (int): Segment length.
            extra_steps (int): Extra steps for n-step reward based credit assignment.
                Should equal n-1 when n steps are used.
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

    @property
    def observation_dtype(self):
        return torch_dtype(self._obs_space.dtype.name)

    @property
    def action_dtype(self):
        return torch_dtype(self._ac_space.dtype.name)

    def _erase(self) -> None:
        obs_shape, ac_shape = self._obs_space.shape, self._ac_space.shape
        self._observations = tc.zeros(
            size=(self._timesteps + 1, *obs_shape),
            dtype=self.observation_dtype,
        )
        self._actions = tc.zeros(
            size=(self._timesteps + 1, *ac_shape),
            dtype=self.action_dtype,
        )
        self._rewards = {
            k: tc.zeros(self._timesteps, dtype=tc.float32)
            for k in self._rew_keys
        }
        self._dones = tc.zeros(self._timesteps, dtype=tc.float32)

    def record(
            self, t: int, o_t: Observation, a_t: Action, r_t: DictReward,
            d_t: bool) -> None:
        """
        Records a timestep of experience to internal experience buffers.

        Args:
            t (int): Integer index for the step to be stored.
            o_t (Observation): Observation.
            a_t (Action): Action.
            r_t (DictReward): Reward dict.
            d_t (bool): Done signal.

        Returns:
            None.
        """
        i = t % self._timesteps
        self._observations[i] = tc.tensor(o_t, dtype=self.observation_dtype)
        self._actions[i] = tc.tensor(a_t, dtype=self.action_dtype)
        for key in self._rew_keys:
            self._rewards[key][i] = tc.tensor(r_t[key]).float()
        self._dones[i] = tc.tensor(d_t).float()

    def record_nexts(self, o_Tp1: Observation, a_Tp1: Action) -> None:
        """
        Records the next observation and next action to rightmost slot of
        internal experience buffers.
        
        Args:
            o_Tp1 (Observation): Next observation.
            a_Tp1 (Action): Next action.

        Returns:
            None.
        """
        self._observations[-1] = tc.tensor(o_Tp1).float()
        self._actions[-1] = tc.tensor(a_Tp1).long()

    def report(self) -> Dict[str, tc.Tensor]:
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
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            policy_net: DDP,
            seg_len: int,
            extra_steps: int):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper instance.
            policy_net (torch.nn.parallel.DistributedDataParallel): DDP-wrapped
                `Agent` instance. Must have 'policy' as a prediction key.
            seg_len (int): Trajectory segment length.
            extra_steps (int): Extra steps for n-step reward based credit assignment.
                Should equal n-1 when n steps are used.
        """
        assert seg_len > 0
        assert extra_steps >= 0

        self._env = env
        self._policy_net = policy_net
        self._seg_len = seg_len
        self._extra_steps = extra_steps

        self._o_t = self._env.reset()
        self._a_t = self._choose_action(self._o_t)
        self._trajectory = Trajectory(
            obs_space=self._env.observation_space,
            ac_space=self._env.action_space,
            rew_keys=self._get_reward_keys(),
            seg_len=self._seg_len,
            extra_steps=self._extra_steps)
        self._metadata_mgr = MetadataManager(
            defaults={
                'ep_len': 0, 'ep_ret': 0., 'ep_len_raw': 0, 'ep_ret_raw': 0.
            })
        _ = self.generate(initial=True)

    def _get_reward_keys(self) -> List[str]:
        if self._env.reward_spec is None:
            msg1 = "Reward spec cannot be None."
            msg2 = "Wrap environment with RewardToDict wrapper to fix."
            raise ValueError(f"{msg1}\n{msg2}")
        return self._env.reward_spec.keys

    def _choose_action(self, o_t: Observation) -> Action:
        # todo(lucaslingle): add support for stateful policies
        # todo(lucaslingle): add support for RL^2/NGU-style inputting
        #    of past rewards as inputs
        predictions = self._policy_net(
            observations=tc.tensor(o_t).unsqueeze(0), predict=['policy'])
        pi_dist_t = predictions.get('policy')
        a_t = pi_dist_t.sample().squeeze(0).detach().numpy()
        return a_t

    def _step_env(self, a_t: Action) -> EnvOutput:
        o_tp1, r_t, done_t, info_t = self._env.step(a_t)
        return o_tp1, r_t, done_t, info_t

    @tc.no_grad()
    def generate(self, initial: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generates a trajectory by stepping the environment.

        Args:
            initial (bool): If True, steps the environment for self._extra_steps
               and saves the results without returning anything. Otherwise,
               the full trajectory is returned.

        Returns:
            Optional[Dict[str, Any]]:
                Maybe a dictionary of trajectory data and metadata, or None.
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
                })
            if done_t:
                self._metadata_mgr.present_done('ep_len', 'ep_ret')

                def was_real_done():
                    if 'ale.lives' in info_t:
                        return info_t['ale.lives'] == 0
                    return True

                if was_real_done():
                    self._metadata_mgr.present_done('ep_len_raw', 'ep_ret_raw')
                o_tp1 = self._env.reset()
            a_tp1 = self._choose_action(o_tp1)
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
            'metadata': self._metadata_mgr.pasts
        }
        self._metadata_mgr.past_done()
        return results
