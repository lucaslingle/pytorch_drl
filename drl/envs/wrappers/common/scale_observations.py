from drl.envs.wrappers.common.abstract import ObservationWrapper
from drl.utils.typing_util import Env


class ScaleObservationsWrapper(ObservationWrapper):
    def __init__(self, env, scale_factor):
        """
        :param env (Env): OpenAI gym environment instance.
        :param scale_factor (float): Scale factor to multiply by.
        """
        super().__init__(env)
        self._scale_factor = scale_factor

    def observation(self, obs):
        return self._scale_factor * obs
