from collections import deque

import numpy as np
import gym

from drl.envs.wrappers.stateless.abstract import Wrapper


class FrameStackWrapper(Wrapper):
    def __init__(self, env, num_stack, lazy=True):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            num_stack (int): Number of frames to stack.
        """
        super().__init__(env)
        self._num_frames = num_stack
        self._lazy = lazy
        self._frames = deque(maxlen=self._num_frames)
        self._set_observation_space()

    def _set_observation_space(self):
        start_shape = self.env.observation_space.shape
        stacked_shape = (*start_shape[:-1], start_shape[-1] * self._num_frames)
        dtype = self.env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=stacked_shape, dtype=dtype)

    def _get_obs(self):
        assert len(self._frames) == self._num_frames
        if not self._lazy:
            return np.concatenate(list(self._frames), axis=-1)
        return LazyFrames(list(self._frames))

    def reset(self, **kwargs):
        obs = self.env.reset()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), rew, done, info


class LazyFrames:
    """
    Helper class to facilitate frame-stacking without a deep copy of frames.
    """
    def __init__(self, frames):
        """
        Args:
            frames (List[np.ndarray]): List of observations.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
