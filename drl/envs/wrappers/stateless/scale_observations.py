"""
Cast and scale observations wrapper.
"""

import numpy as np
import gym

from drl.envs.wrappers.stateless.abstract import ObservationWrapper


class ScaleObservationsWrapper(ObservationWrapper):
    def __init__(self, env, scale_factor):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            scale_factor (float): Scale factor to multiply by.
        """
        super().__init__(env)
        self._scale_factor = scale_factor
        self._set_observation_space()

    def _set_observation_space(self):
        space = self.observation_space
        low = space.low * self._scale_factor
        high = space.high * self._scale_factor
        shape = space.shape
        new_space = gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=np.float32)
        self.observation_space = new_space

    def observation(self, obs):
        return self._scale_factor * obs.astype(np.float32)
