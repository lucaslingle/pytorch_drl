from drl.envs.wrappers.common.abstract import ObservationWrapper
from drl.utils.typing_util import Env


class ScaleObservationsWrapper(ObservationWrapper):
    def __init__(self, env, scale_factor):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            scale_factor (float): Scale factor to multiply by.
        """
        super().__init__(env)
        self._scale_factor = scale_factor
        self._run_checks()

    def _run_checks(self):
        cond = str(self.env.observation_space.dtype) == 'float32'
        if not cond:
            msg = "Attempted to wrap env with non-float32 obs dtype."
            raise ValueError(msg)

    def observation(self, obs):
        return self._scale_factor * obs
