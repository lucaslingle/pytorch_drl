"""
Frame skipping and pooling wrapper.
"""

from typing import Union
from collections import deque

import numpy as np
import gym

from drl.envs.wrappers.stateless.abstract import Wrapper
from drl.utils.typing import ActionType, EnvStepOutput


class MaxAndSkipWrapper(Wrapper):
    """
    Max and skip wrapper. Skips frames and optionally applies max-pooling
    to the last n frames.
    """
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            num_skip: int,
            apply_maxpool: bool,
            depth_maxpool: int = 2):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            num_skip (int): Number of frames to skip over.
            apply_maxpool (bool): Apply max-pooling?
            depth_maxpool (int): Number of frames to apply maxpooling to.
                Default: 2.
        """
        super().__init__(env)
        self._num_skip = num_skip
        self._apply_maxpool = apply_maxpool
        self._obs_buffer = deque(maxlen=depth_maxpool)
        self._run_checks()

    def _run_checks(self):
        cond1 = isinstance(self._num_skip, int)
        cond2 = self._num_skip > 0
        if not cond1:
            msg = "Number of frames to skip must be an integer."
            raise TypeError(msg)
        if not cond2:
            msg = "Number of frames to skip must be greater than zero."
            raise ValueError(msg)

    def step(self, action: ActionType) -> EnvStepOutput:
        total_reward = 0.
        for k in range(self._num_skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        if self._apply_maxpool:
            obs = np.array(list(self._obs_buffer)).max(axis=0)
        return obs, total_reward, done, info
