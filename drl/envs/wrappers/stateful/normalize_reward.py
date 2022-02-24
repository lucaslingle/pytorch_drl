"""
Reward normalization wrapper.
"""

from typing import Union, Optional, Dict, Any, Mapping, List
import collections

import torch as tc
import gym

from drl.envs.wrappers.stateless.abstract import Wrapper, RewardSpec
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.algos.common import global_mean
from drl.utils.types import Checkpointable, Action, EnvOutput

RewardAndDone = collections.namedtuple('RewardAndDone', ['reward', 'done'])


class ReturnAcc(tc.nn.Module):
    def __init__(self, gamma, clip_low, clip_high, use_dones):
        super().__init__()
        self._gamma = gamma
        self._use_dones = use_dones
        self._current_ep_rewards = []
        self._normalizer = Normalizer([], clip_low, clip_high)
        self._trace_length = int(5 / (1 - gamma))

    @property
    def steps(self) -> tc.Tensor:
        return self._normalizer.steps

    @steps.setter
    def steps(self, value: tc.Tensor) -> None:
        self._normalizer.steps = value

    @property
    def moment1(self) -> tc.Tensor:
        return self._normalizer.moment1

    @moment1.setter
    def moment1(self, value: tc.Tensor) -> None:
        self._normalizer.moment1 = value

    @property
    def moment2(self) -> tc.Tensor:
        return self._normalizer.moment2

    @moment2.setter
    def moment2(self, value: tc.Tensor) -> None:
        self._normalizer.moment2 = value

    @property
    def trace_length(self) -> int:
        return self._trace_length

    @trace_length.setter
    def trace_length(self, value: int) -> None:
        self._trace_length = value

    def _unload(self) -> List[float]:
        returns = [0. for _ in range(2 * self._trace_length + 1)]
        for t in reversed(range(2 * self._trace_length)):
            rd = self._current_ep_rewards[t]
            r_t, d_t = rd.reward, rd.done
            mask = float(1. - d_t) if self._use_dones else 1.
            returns[t] = r_t + mask * self._gamma * returns[t + 1]
        self._current_ep_rewards = self._current_ep_rewards[self.trace_length:]
        return returns[0:self.trace_length]

    def _update_from_returns(self, returns: List[float]) -> None:
        returns = tc.tensor(returns)
        ep_steps = returns.shape[0]
        steps = self._normalizer.steps + ep_steps

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
        rd = RewardAndDone(reward=r_t, done=d_t)
        self._current_ep_rewards.append(rd)
        if len(self._current_ep_rewards) >= 2 * self._trace_length:
            returns = self._unload()
            self._update_from_returns(returns)

    def forward(self, r_t, shift=False, scale=True) -> float:
        if self.steps.item() == 0:
            return 0.
        return self._normalizer(r_t, shift=shift, scale=scale).item()


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
        self._synced_normalizer = ReturnAcc(gamma, -10., 10., use_dones)
        self._unsynced_normalizer = ReturnAcc(gamma, -10., 10., use_dones)
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
    def trace_length(self) -> int:
        assert self._unsynced_normalizer.trace_length == \
               self._synced_normalizer.trace_length
        return self._synced_normalizer.trace_length

    @trace_length.setter
    def trace_length(self, value: int) -> None:
        self._unsynced_normalizer.trace_length = value
        self._synced_normalizer.trace_length = value

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
        reward = rew[self._key] if isinstance(rew, dict) else rew
        reward = tc.tensor(reward).float()
        normalized = self._synced_normalizer(reward.unsqueeze(0))
        self._unsynced_normalizer.update(reward, done)
        if isinstance(rew, dict):
            rew[self._key] = normalized
        else:
            rew = {'extrinsic_raw': reward, 'extrinsic': normalized}
        return obs, rew, done, info

    def learn(
        self,
        minibatch: Mapping[str, tc.Tensor],
        **kwargs: Mapping[str, Any],
    ) -> None:
        self._sync_normalizers_global()
        self._sync_normalizers_local()
