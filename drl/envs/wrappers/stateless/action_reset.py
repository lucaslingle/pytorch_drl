"""
Action reset wrapper.
"""

from typing import Union, List, Any, Mapping

import gym

from drl.envs.wrappers.stateless.abstract import Wrapper
from drl.utils.typing import Action, Observation


class ActionResetWrapper(Wrapper):
    """
    Action reset wrapper. Takes actions on reset for environments that are
    frozen until the action sequence occurs.
    """
    def __init__(
            self, env: Union[gym.core.Env, Wrapper],
            action_sequence: List[Action]):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            action_sequence (List[Action]): List of actions to perform.
        """
        super().__init__(env)
        self._action_sequence = action_sequence

    @property
    def action_sequence(self):
        return self._action_sequence

    def reset(self, **kwargs: Mapping[str, Any]) -> Observation:
        """
        Resets environment and takes actions listed in `self.action_sequence`.
        """
        obs = self.env.reset(**kwargs)
        for a in self.action_sequence:
            obs, _, done, _ = self.env.step(a)
            if done:
                _ = self.env.reset(**kwargs)
        return obs
