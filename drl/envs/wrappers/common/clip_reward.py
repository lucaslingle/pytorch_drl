from drl.envs.wrappers.common.abstract import RewardWrapper
from drl.utils.typing_util import Env


class ClipRewardWrapper(RewardWrapper):
    def __init__(self, env, low, high):
        """
        :param env (Env): OpenAI gym environment instance.
        :param low (float): Minimum value for clipped reward.
        :param high (float): Maximum value for clipped reward.
        """
        super().__init__(env)
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
        self.reward_range = (self._low, self._high)

    def reward(self, reward):
        return max(self._low, min(reward, self._high))