from drl.envs.wrappers.common.abstract import Wrapper


ATARI_NOOP_ACTION = 0
ATARI_MIN_NOOPS = 30
ATARI_MAX_NOOPS = 30


class NoopResetWrapper(Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    """
    def __init__(self, env, noop_action, noop_min, noop_max):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        :param noop_action (int): No-op action.
        :param noop_min (int): Minimum number of no-op actions to take.
        :param noop_max (int): Maximum number of no-op actions to take.
        """
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[noop_action] == 'NOOP'
        self._noop_action = noop_action
        self._noop_min = noop_min
        self._noop_max = noop_max

    def reset(self, **kwargs):
        """
        Takes a random number of no-op actions between
        self._noop_min and self._noop_max.
        """
        obs = self.env.reset(**kwargs)
        low, high = self._noop_min, self._noop_max+1
        noops = self.unwrapped.np_random.randint(low, high)
        if noops == 0:
            return obs
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self._noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
