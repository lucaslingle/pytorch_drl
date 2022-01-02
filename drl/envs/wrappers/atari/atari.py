import drl.envs.wrappers.common as ws
import drl.envs.wrappers.atari.constants as acs


class AtariWrapper(ws.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        assert 'NoFrameskip' in env.spec.id
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self.env = self._build()

    def _build(self):
        env = ws.NoopResetWrapper(
            env=self.env,
            noop_action=acs.NOOP_ACTION,
            noop_min=acs.MIN_RESET_NOOPS,
            noop_max=acs.MAX_RESET_NOOPS)
        env = ws.MaxAndSkipWrapper(
            env=env,
            num_skip=acs.NUM_SKIP,
            apply_max=acs.APPLY_MAX)
        if self._max_episode_steps is not None:
            env = ws.TimeLimitWrapper(
                env=env,
                max_episode_steps=self._max_episode_steps)
        return env
