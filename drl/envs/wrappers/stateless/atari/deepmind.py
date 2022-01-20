import drl.envs.wrappers.stateless as ws
import drl.envs.wrappers.stateless.atari.constants as acs


class DeepmindWrapper(ws.Wrapper):
    def __init__(
            self,
            env,
            episode_life=True,
            clip_rewards=True,
            scale=True,
            frame_stack=True,
            lazy=True
    ):
        """
        Configure environment for DeepMind-style Atari.

        Args:
            env (Env): OpenAI gym environment instance.
            episode_life (bool): Use EpisodicLifeWrapper?
            clip_rewards (bool): Use ClipRewardWrapper?
            scale (bool): Use ScaleObservationsWrapper?
            frame_stack (bool): Use FrameStackWrapper?
            lazy (bool): Use LazyFrames in FrameStackWrapper?
        """
        super().__init__(env)
        self._episode_life = episode_life
        self._clip_rewards = clip_rewards
        self._scale = scale
        self._frame_stack = frame_stack
        self._lazy = lazy
        self.env = self._build()

    def _build(self):
        env = ws.ResizeWrapper(
            env=self.env, width=acs.TARGET_WIDTH, height=acs.TARGET_HEIGHT,
            grayscale=acs.USE_GRAYSCALE)

        if self._episode_life:
            env = ws.EpisodicLifeWrapper(
                env=env, lives_fn=acs.LIVES_FN, noop_action=acs.NOOP_ACTION)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = ws.ActionResetWrapper(
                env=env, action_sequence=acs.ACTION_RESET_SEQUENCE)
        if self._clip_rewards:
            env = ws.ClipRewardWrapper(
                env=env, low=acs.REWARD_CLIP_LOW, high=acs.REWARD_CLIP_HIGH,
                key='extrinsic')
        if self._scale:
            env = ws.ScaleObservationsWrapper(
                env=env, scale_factor=acs.SCALE_FACTOR)
        if self._frame_stack:
            env = ws.FrameStackWrapper(
                env=env, num_stack=acs.NUM_STACK, lazy=self._lazy)
        return env
