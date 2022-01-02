from drl.envs.wrappers.common.abstract import Wrapper


class NoopResetWrapper(Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    """
    def __init__(self, env, noop_action, noop_min, noop_max):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            noop_action (int): No-op action.
            noop_min (int): Minimum number of no-op actions to take.
            noop_max (int): Maximum number of no-op actions to take.
        """
        super().__init__(env)
        self._noop_action = noop_action
        self._noop_min = noop_min
        self._noop_max = noop_max
        self._run_checks()

    def _run_checks(self):
        meanings = self.env.unwrapped.get_action_meanings()
        cond = meanings[self._noop_action] == 'NOOP'
        if not cond:
            msg = "Chosen no-op action does not have meaning 'NOOP'."
            raise ValueError(msg)

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
