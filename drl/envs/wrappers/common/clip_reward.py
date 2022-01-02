from drl.envs.wrappers.common.abstract import RewardWrapper


class ClipRewardWrapper(RewardWrapper):
    def __init__(self, env, low, high, key=None):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            low (float): Minimum value for clipped reward.
            high (float): Maximum value for clipped reward.
        """
        super().__init__(env)
        self._low = low
        self._high = high
        self._key = key
        self._run_checks()

    def _run_checks(self):
        cond = self._low < self._high
        if not cond:
            msg = "Low value must be less than high value."
            raise ValueError(msg)

    def reward(self, reward):
        if self._key:
            if not isinstance(reward, dict):
                msg = "Keyed ClipRewardWrapper expected reward to be a dict."
                raise TypeError(msg)
            selected_reward = reward[self._key]
            reward[self._key] = max(self._low, min(selected_reward, self._high))
            return reward
        return max(self._low, min(reward, self._high))
