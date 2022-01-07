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
        prev_obs, total_reward, done = None, 0., False
        for _ in range(self._num_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            prev_obs = obs
            if done:
                break
        obs = np.maximum(prev_obs, obs) if self._apply_max else obs
        return obs, total_reward, done, info
