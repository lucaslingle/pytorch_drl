import drl.envs.wrappers.stateless as ws
import drl.envs.wrappers.stateless.atari.constants as acs


class AtariWrapper(ws.Wrapper):
    def __init__(self, env, use_noop, use_sticky_actions, max_episode_frames=None):
        assert 'NoFrameskip' in env.spec.id
        super().__init__(env)
        self._use_noop = use_noop
        self._use_sticky_actions = use_sticky_actions
        self._max_episode_frames = max_episode_frames
        self.env = self._build()

    def _build(self):
        if self._max_episode_frames is not None:
            env = ws.TimeLimitWrapper(
                env=self.env,
                max_episode_steps=self._max_episode_frames)
        if self._use_noop:
            env = ws.NoopResetWrapper(
                env=self.env,
                noop_action=acs.NOOP_ACTION,
                noop_min=acs.MIN_RESET_NOOPS,
                noop_max=acs.MAX_RESET_NOOPS)
        if self._use_sticky_actions:
            env = ws.StickyActionsWrapper(
                env=self.env,
                stick_prob=acs.STICK_PROB)
        env = ws.MaxAndSkipWrapper(
            env=env,
            num_skip=acs.NUM_SKIP,
            apply_max=acs.APPLY_MAX)
        env = ws.RewardToDictWrapper(env)
        return env
