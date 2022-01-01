import torch as tc
import numpy as np

from drl.envs.wrappers.common.abstract import ObservationWrapper


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

    def observation(self, obs):
        return tc.tensor(obs.astype(np.int32)).float()
