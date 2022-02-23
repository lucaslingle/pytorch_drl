"""
Reward normalization wrapper.
"""

from typing import Union, Optional, Dict, Any, Mapping, List

import torch as tc
import gym

from drl.envs.wrappers.stateless.abstract import Wrapper, RewardSpec
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.algos.common import global_mean
from drl.utils.types import Checkpointable, Action, EnvOutput


class _ReturnAcc(tc.nn.Module):
    def __init__(self, gamma, clip_low, clip_high, use_dones):
        super().__init__()
        self._gamma = gamma
        self._use_dones = use_dones
        self._current_ep_rewards = []
        self._normalizer = Normalizer([], clip_low, clip_high)

    @property
    def steps(self):
        return self._normalizer.steps

    @steps.setter
    def steps(self, value):
        self._normalizer.steps = value

    @property
    def moment1(self):
        return self._normalizer.moment1

    @moment1.setter
    def moment1(self, value):
        self._normalizer.moment1 = value

    @property
    def moment2(self):
        return self._normalizer.moment2

    @moment2.setter
    def moment2(self, value):
        self._normalizer.moment2 = value

    @property
    def trace_length(self) -> Optional[int]:
        if self._use_dones:
            return None
        return int(5 / (1 - self._gamma))

    def _unload_terminating(self) -> List[float]:
        returns = list()
        while len(self._current_ep_rewards) > 0:
            r_t = self._current_ep_rewards.pop()
            if len(returns) == 0:
                R_t = r_t
            else:
                R_tp1 = returns[-1]
                R_t = r_t + self._gamma * R_tp1
            returns.append(R_t)
        return returns[::-1]

    def _unload_continuing(self) -> List[float]:
        returns = list()
        for t in range(2 * self.trace_length):
            r_t = self._current_ep_rewards[2 * self.trace_length - 1 - t]
            if len(returns) == 0:
                R_t = r_t
            else:
                R_tp1 = returns[-1]
                R_t = r_t + self._gamma * R_tp1
            returns.append(R_t)
        self._current_ep_rewards = self._current_ep_rewards[self.trace_length:]
        return returns[::-1][0:self.trace_length]

    def update_from_returns(self, returns: List[float]):
        ep_steps = len(returns)
        steps = self._normalizer.steps + ep_steps

        returns = tc.tensor(returns)

        moment1 = self._normalizer.moment1
        moment1 *= ((steps - ep_steps) / steps)
        ep_ret_mean = tc.mean(returns)
        moment1 += (ep_steps / steps) * ep_ret_mean

        moment2 = self._normalizer.moment2
        moment2 *= ((steps - ep_steps) / steps)
        ep_ret_var = tc.mean(tc.square(returns))
        moment2 += (ep_steps / steps) * ep_ret_var

        self._normalizer.steps = steps
        self._normalizer.moment1 = moment1
        self._normalizer.moment2 = moment2

    def update(self, r_t: float, d_t: bool) -> None:
        self._current_ep_rewards.append(r_t)
        if self._use_dones:
            if d_t:
                returns = self._unload_terminating()
                self.update_from_returns(returns)
        if not self._use_dones:
            if len(self._current_ep_rewards) >= 2 * self.trace_length:
                returns = self._unload_continuing()
                self.update_from_returns(returns)

    def forward(self, r_t, shift=False, scale=True) -> tc.Tensor:
        if self.steps == 0:
            return tc.zeros_like(r_t)
        return self._normalizer(r_t, shift=shift, scale=scale)


class NormalizeRewardWrapper(TrainableWrapper):
    """
    Reward normalization wrapper.
    """
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            gamma: float,
            world_size: int,
            use_dones: bool,
            key: Optional[str] = None):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            gamma (float): Discount factor.
            world_size (int): Number of processes.
            use_dones (bool): Truncate returns at episode boundaries?
            key (Optional[str]): Optional reward key.
        """
        super().__init__(env)
        self._synced_normalizer = _ReturnAcc(gamma, -10, 10, use_dones)
        self._unsynced_normalizer = _ReturnAcc(gamma, -10, 10, use_dones)
        self._key = key
        self._world_size = world_size
        self._reward_spec = self._get_reward_spec()

    def _get_reward_spec(self) -> RewardSpec:
        def spec_exists():
            if isinstance(self.env, Wrapper):
                return self.env.reward_spec is not None
            return False

        if not spec_exists():
            keys = ['extrinsic_raw', 'extrinsic']
            return RewardSpec(keys)
        else:
            return self.env.reward_spec

    def _sync_normalizers_global(self) -> None:
        self._synced_normalizer.steps = global_mean(
            self._unsynced_normalizer.steps, self._world_size)
        self._synced_normalizer.moment1 = global_mean(
            self._unsynced_normalizer.moment1, self._world_size)
        self._synced_normalizer.moment2 = global_mean(
            self._unsynced_normalizer.moment2, self._world_size)

    def _sync_normalizers_local(self) -> None:
        self._unsynced_normalizer.steps = self._synced_normalizer.steps
        self._unsynced_normalizer.moment1 = self._synced_normalizer.moment1
        self._unsynced_normalizer.moment2 = self._synced_normalizer.moment2

    @property
    def checkpointables(self) -> Dict[str, Checkpointable]:
        checkpoint_dict = self.env.checkpointables
        checkpoint_dict.update({'reward_normalizer': self._synced_normalizer})
        return checkpoint_dict

    def step(self, ac: Action) -> EnvOutput:
        if self._unsynced_normalizer.steps < self._synced_normalizer.steps:
            self._sync_normalizers_local()
        obs, rew, done, info = self.env.step(ac)
        if self._key:
            if self._key == 'extrinsic_raw':
                msg = "The key 'extrinsic_raw' must be preserved for logging."
                raise ValueError(msg)
            if not isinstance(rew, dict):
                msg = "Got non-keyed reward with keyed normalization wrapper."
                raise TypeError(msg)
        else:
            if isinstance(rew, dict):
                msg = "Got keyed reward w/ non-keyed normalization wrapper."
                raise TypeError(msg)
        reward = rew if not isinstance(rew, dict) else rew[self._key]
        reward = tc.tensor(reward).float()
        normalized = self._synced_normalizer(reward.unsqueeze(0)).item()
        self._unsynced_normalizer.update(reward, done)
        if isinstance(rew, dict):
            rew[self._key] = normalized
        else:
            rew = {'extrinsic_raw': reward, 'extrinsic': normalized}
        return obs, rew, done, info

    def learn(
        self,
        mb: Mapping[str, tc.Tensor],
        **kwargs: Mapping[str, Any],
    ) -> None:
        self._sync_normalizers_global()
        self._sync_normalizers_local()
