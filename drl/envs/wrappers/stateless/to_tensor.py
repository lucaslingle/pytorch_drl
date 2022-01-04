import torch as tc
import numpy as np
import gym

from drl.envs.wrappers.stateless.abstract import ObservationWrapper


class ToTensorWrapper(ObservationWrapper):
    """
    Observation wrapper to transform observations into tensors.
    """
    def __init__(self, env):
        """
        Args:
            env (Env): OpenAI gym environment instance.
        """
        super().__init__(env)
        self._run_checks()
        self._set_observation_space()

    def _run_checks(self):
        supported_dtypes = ['uint8', 'int32', 'int64', 'float32', 'float64']
        dtype = str(self.env.observation_space.dtype)
        cond = dtype in supported_dtypes
        if not cond:
            msg = f"Attempted to wrap env with non-supported dtype {dtype}."
            raise TypeError(msg)

    def _set_observation_space(self):
        space = self.env.observation_space
        low, high, shape = space.low, space.high, space.shape
        dtype = np.float32
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=dtype)

    def observation(self, obs):
        return tc.tensor(obs).float()
