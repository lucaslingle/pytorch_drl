import torch as tc

from drl.envs.wrappers.common.abstract import ObservationWrapper


class ToTensorWrapper(ObservationWrapper):
    """
    Observation wrapper to transform observations into tensors.
    """
    def __init__(self, env):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        """
        super().__init__(env)

    def observation(self, obs):
        return tc.tensor(obs).float()
