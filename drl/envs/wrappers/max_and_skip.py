import numpy as np

from drl.envs.wrappers.abstract import Wrapper


ATARI_NUM_SKIP = 4
ATARI_APPLY_MAX = True


class MaxAndSkipWrapper(Wrapper):
    """
    Skip frames and optionally apply max-pooling to the last two frames.
    """
    def __init__(self, env, num_skip, apply_max):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        :param num_skip (int): Number of frames to skip over.
        :param apply_max (bool): Apply max-pooling to the last two frames?
            This can reduce artifacts in some environments, such as the ALE.
        """
        super().__init__(self, env)
        self._num_skip = num_skip
        self._apply_max = apply_max

    def step(self, action):
        prev_obs, total_reward, done = None, 0., False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            prev_obs = obs
            if done:
                break
        obs = np.maximum(prev_obs, obs) if self._apply_max else obs
        return obs, total_reward, done, info
