from drl.envs.wrappers.stateless.abstract import RewardWrapper, RewardSpec


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
        self._reward_spec = self._get_reward_spec()

    def _run_checks(self):
        cond = self._low < self._high
        if not cond:
            msg = "Low value must be less than high value."
            raise ValueError(msg)

    def _get_reward_spec(self):
        parent_reward_spec = self.env.reward_spec
        if parent_reward_spec is None:
            reward_keys = ['extrinsic_raw', 'extrinsic']
        else:
            reward_keys = parent_reward_spec.keys
        return RewardSpec(keys=reward_keys)

    def reward(self, reward):
        if self._key:
            assert self._key != 'extrinsic_raw', 'Must be preserved for logging'
            if not isinstance(reward, dict):
                msg = "Keyed ClipRewardWrapper expected reward to be a dict."
                raise TypeError(msg)
            selected_reward = reward[self._key]
            reward[self._key] = max(self._low, min(selected_reward, self._high))
            return reward
        else:
            reward_clipped = max(self._low, min(reward, self._high))
            return {'extrinsic_raw': reward, 'extrinsic': reward_clipped}
