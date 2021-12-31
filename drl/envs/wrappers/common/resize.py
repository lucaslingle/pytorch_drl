import numpy as np
import gym
import cv2

from drl.envs.wrappers.common.abstract import ObservationWrapper

ATARI_TARGET_HEIGHT = 84
ATARI_TARGET_WIDTH = 84
ATARI_USE_GRAYSCALE = True


class ResizeWrapper(ObservationWrapper):
    """
    Resize frames and optionally convert to grayscale.
    """
    def __init__(self, env, width, height, grayscale):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        :param width (int): Target image height.
        :param height (int): Target image height.
        :param grayscale (bool): Convert to grayscale?
        """
        super().__init__(env)
        _ = cv2.ocl.setUseOpenCL(False)
        self._width = width
        self._height = height
        self._grayscale = grayscale

        self._run_checks()
        self._set_observation_space()

    def _run_checks(self):
        starting_space = self.observation_space
        cond1 = starting_space.dtype == np.uint8
        cond2 = len(starting_space.shape) == 3
        if not cond1:
            raise ValueError("Required starting space dtype to be np.uint8.")
        if not cond2:
            raise ValueError("Required starting space shape to have length 3.")

    def _set_observation_space(self):
        num_colors = 1 if self._grayscale else 3
        shape = (self._height, self._width, num_colors)
        new_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.observation_space = new_space

    def observation(self, obs):
        frame = obs
        target_shape = (self._width, self._height)

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, target_shape, interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        obs = frame
        return obs
