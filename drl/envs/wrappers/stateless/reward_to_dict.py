"""
Reward to dictionary wrapper.
"""

from typing import Union

import gym

from drl.envs.wrappers.stateless.abstract import (
    Wrapper, RewardWrapper, RewardSpec)
from drl.utils.typing import Reward, DictReward


class RewardToDictWrapper(RewardWrapper):
    """
    Reward to dictionary wrapper. Also creates a reward spec for the wrapped environment.
    """
    def __init__(self, env: Union[gym.core.Env, Wrapper]):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
        """
        super().__init__(env)
        self._reward_spec = self._get_reward_spec()

    def _get_reward_spec(self) -> RewardSpec:
        parent_reward_spec = self.env.reward_spec
        if parent_reward_spec is None:
            reward_keys = ['extrinsic_raw', 'extrinsic']
        else:
            reward_keys = parent_reward_spec.keys
        return RewardSpec(keys=reward_keys)

    def reward(self, reward: Reward) -> DictReward:
        if isinstance(reward, dict):
            return reward
        if isinstance(reward, float):
            return {'extrinsic_raw': reward, 'extrinsic': reward}
        raise TypeError("Unsupported reward type.")
