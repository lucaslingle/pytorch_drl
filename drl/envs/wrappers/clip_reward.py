import numpy as np
import gym

from drl.envs.wrappers.abstract import RewardWrapper

ATARI_LOW = -1.0
ATARI_HIGH = 1.0


class ClipRewardWrapper(RewardWrapper):
    def __init__(self, env, low, high):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        :param low (float): Minimum value for clipped reward.
        :param high (float): Maximum value for clipped reward.
        """
        super().__init__(self, env)
        self._low = low
        self._high = high

        self._run_checks()
        self._set_reward_range()

    def _run_checks(self):
        cond = self._low < self._high
        if not cond:
            msg = "Low value must be less than high value."
            raise ValueError(msg)

    def _set_reward_range(self):
        self.reward_range = gym.spaces.Box(
            low=self._low, high=self._high, shape=(1,), dtype=np.float32)

    def reward(self, reward):
        return max(self._low, min(reward, self._high))