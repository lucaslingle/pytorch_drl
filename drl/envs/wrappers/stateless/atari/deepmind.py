from typing import Union

import gym

import drl.envs.wrappers.stateless as ws
import drl.envs.wrappers.stateless.atari.constants as acs


class DeepmindWrapper(ws.Wrapper):
    """
    Deepmind-style Atari wrapper.
    """
    def __init__(
            self,
            env: Union[gym.core.Env, ws.Wrapper],
            episode_life: bool = True,
            clip_rewards: bool = True,
            scale: bool = True,
            frame_stack: bool = True,
            lazy: bool = False):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            episode_life (bool): Use EpisodicLifeWrapper? Default: True.
            clip_rewards (bool): Use ClipRewardWrapper? Default: True.
            scale (bool): Use ScaleObservationsWrapper? Default: True.
            frame_stack (bool): Use FrameStackWrapper? Default: True.
            lazy (bool): Use LazyFrames in FrameStackWrapper? Default: False.
        """
        super().__init__(env)
        self._episode_life = episode_life
        self._clip_rewards = clip_rewards
        self._scale = scale
        self._frame_stack = frame_stack
        self._lazy = lazy
        self.env = self._build()

    def _build(self) -> ws.Wrapper:
        env = ws.ResizeObservationsWrapper(
            env=self.env,
            width=acs.TARGET_WIDTH,
            height=acs.TARGET_HEIGHT,
            grayscale=acs.USE_GRAYSCALE)

        if self._episode_life:
            env = ws.EpisodicLifeWrapper(
                env=env, lives_fn=acs.LIVES_FN, noop_action=acs.NOOP_ACTION)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = ws.ActionResetWrapper(
                env=env, action_sequence=acs.ACTION_RESET_SEQUENCE)
        if self._clip_rewards:
            env = ws.ClipRewardWrapper(
                env=env,
                low=acs.REWARD_CLIP_LOW,
                high=acs.REWARD_CLIP_HIGH,
                key='extrinsic')
        if self._scale:
            env = ws.ScaleObservationsWrapper(
                env=env, scale_factor=acs.SCALE_FACTOR)
        if self._frame_stack:
            env = ws.FrameStackWrapper(
                env=env, num_stack=acs.NUM_STACK, lazy=self._lazy)
        return env
