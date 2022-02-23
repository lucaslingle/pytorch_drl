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
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            stick_prob: float,
            noop_action: Action) -> None:
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            stick_prob (float): Number between 0 and 1.
        """
        assert 0. <= stick_prob <= 1.
        super().__init__(env)
        self._stick_prob = stick_prob
        self._last_action = noop_action

    @property
    def last_action(self) -> Action:
        return self._last_action

    @last_action.setter
    def last_action(self, action: Action) -> None:
        self._last_action = action

    def _sample_uniform(self):
        return self.unwrapped.np_random.uniform(low=0., high=1.)

    def logic(self, action: Action, u: float) -> Action:
        """
        Deterministically replaces action with previous action if
        the input variable u is less than `self._stick_prob`.

        Args:
            action (Action): Input action.
            u (float):

        Returns:
            Action: Updated action to be performed.
        """
        if u < self._stick_prob:
            return self._last_action
        self._last_action = action
        return action

    def action(self, action: Action) -> Action:
        """
        Randomly replaces action with previous action using probability
            `self._stick_prob`.

        Args:
            action (Action): Input action.

        Returns:
            Action: Updated action to be performed.
        """
        u = self._sample_uniform()
        return self.logic(action, u)
