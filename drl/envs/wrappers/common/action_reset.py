from typing import List, Dict, Any

from drl.envs.wrappers.common.abstract import Wrapper
from drl.utils.types_util import Env


class ActionResetWrapper(Wrapper):
    """
    Take action on reset for environments that are fixed until
    action sequence occurs.
    """
    def __init__(self, env, action_sequence):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            action_sequence (List[int]): List of actions to perform.
        """
        super().__init__(env)
        self._action_sequence = action_sequence

    def reset(self, **kwargs: Dict[str, Any]):
        """
        Resets environment and takes actions listed in self._action_sequence.
        """
        obs = self.env.reset(**kwargs)
        for a in self._action_sequence:
            obs, _, done, _ = self.env.step(a)
            if done:
                _ = self.env.reset(**kwargs)
        return obs
