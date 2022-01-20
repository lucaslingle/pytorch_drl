from drl.envs.wrappers.stateless.abstract import RewardWrapper, RewardSpec


class RewardToDictWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._reward_spec = self._get_reward_spec()

    def _get_reward_spec(self):
        parent_reward_spec = self.env.reward_spec
        if parent_reward_spec is None:
            reward_keys = ['extrinsic_raw', 'extrinsic']
        else:
            reward_keys = parent_reward_spec.keys
        return RewardSpec(keys=reward_keys)

    def reward(self, reward):
        if isinstance(reward, dict):
            return reward
        if isinstance(reward, float):
            return {'extrinsic_raw': reward, 'extrinsic': reward}
        raise TypeError("Unsupported reward type.")
