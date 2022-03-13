"""
Cast and scale observations wrapper.
"""

from typing import Union

import numpy as np
import gym

from drl.envs.wrappers.stateless.abstract import Wrapper, ObservationWrapper
from drl.utils.types import Observation


class ScaleObservationsWrapper(ObservationWrapper):
    def __init__(self, env: Union[gym.core.Env, Wrapper], scale_factor: float):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            scale_factor (float): Scale factor to multiply by.
        """
        super().__init__(env)
        self._scale_factor = scale_factor
        self._set_observation_space()

    def _set_observation_space(self) -> None:
        space = self.observation_space
        low = space.low * self._scale_factor
        high = space.high * self._scale_factor
        shape = space.shape
        new_space = gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=np.float32)
        self.observation_space = new_space

    def observation(self, obs: Observation) -> Observation:
        return self._scale_factor * obs.astype(np.float32)
