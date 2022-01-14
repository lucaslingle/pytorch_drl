from collections import deque

import numpy as np

from drl.envs.wrappers.stateless.abstract import Wrapper


class MaxAndSkipWrapper(Wrapper):
    """
    Skip frames and optionally apply max-pooling to the last two frames.
    """
    def __init__(self, env, num_skip, apply_max):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            num_skip (int): Number of frames to skip over.
            apply_max (bool): Apply max-pooling to the last two frames?
                This can reduce artifacts in some environments, such as the ALE.
        """
        super().__init__(env)
        self._num_skip = num_skip
        self._apply_max = apply_max
        self._obs_buffer = deque(maxlen=2)
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

    def step(self, action):
        total_reward = 0.
        for k in range(self._num_skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        if self._apply_max:
            obs = np.array(list(self._obs_buffer)).max(axis=0)
        return obs, total_reward, done, info
