"""
Sticky actions wrapper.
"""

from typing import Union

import gym

from drl.envs.wrappers.stateless.abstract import Wrapper, ActionWrapper
from drl.utils.types import Action


class StickyActionsWrapper(ActionWrapper):
    """
    Sticky actions wrapper.

    Reference:
        M. Machado et al., 2017 -
           'Revisiting the Arcade Learning Environment: Evaluation Protocols
            and Open Problems for General Agents'
    """
    def __init__(self, env: Union[gym.core.Env, Wrapper], stick_prob: float):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            stick_prob (float): Number between 0 and 1.
        """
        assert 0. <= stick_prob <= 1.
        super().__init__(env)
        self._stick_prob = stick_prob
        self._last_action = 0

    def action(self, action: Action) -> Action:
        u = self.unwrapped.np_random.uniform(low=0., high=1.)
        if u < self._stick_prob:
            return self._last_action
        self._last_action = action
        return action
