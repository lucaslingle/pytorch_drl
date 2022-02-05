"""
Atari wrapper.
"""

from typing import Optional

import gym

import drl.envs.wrappers.stateless as ws
import drl.envs.wrappers.stateless.atari.constants as acs


class AtariWrapper(ws.Wrapper):
    """
    Atari wrapper.
    """
    def __init__(
            self,
            env: gym.core.Env,
            max_episode_frames: Optional[int] = None,
            use_noop: bool = True,
            use_sticky_actions: bool = False,
            use_frameskip: bool = True):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            max_episode_frames (Optional[int]): Max episode frames. Default: None.
            use_noop: Use no-op resets? Default: True.
            use_sticky_actions: Use sticky actions? Default: False.
            use_frameskip: Use frame skipping? Default: True.
        """
        assert 'NoFrameskip' in env.spec.id
        super().__init__(env)
        self._max_episode_frames = max_episode_frames
        self._use_noop = use_noop
        self._use_sticky_actions = use_sticky_actions
        self._use_frameskip = use_frameskip
        self.env = self._build()

    def _build(self) -> ws.Wrapper:
        env = self.env
        if self._max_episode_frames is not None:
            env = ws.TimeLimitWrapper(
                env=self.env, max_episode_steps=self._max_episode_frames)
        if self._use_noop:
            env = ws.NoopResetWrapper(
                env=self.env,
                noop_action=acs.NOOP_ACTION,
                noop_min=acs.MIN_RESET_NOOPS,
                noop_max=acs.MAX_RESET_NOOPS)
        if self._use_sticky_actions:
            env = ws.StickyActionsWrapper(
                env=self.env, stick_prob=acs.STICK_PROB)
        if self._use_frameskip:
            env = ws.MaxAndSkipWrapper(
                env=env,
                num_skip=acs.NUM_SKIP,
                apply_maxpool=acs.APPLY_MAXPOOL,
                depth_maxpool=acs.DEPTH_MAXPOOL)
        env = ws.RewardToDictWrapper(env)
        return env
