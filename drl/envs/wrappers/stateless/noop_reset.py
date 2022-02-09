"""
Random no-ops after reset wrapper.
"""

from typing import Union, Mapping, Any

import gym

from drl.envs.wrappers.stateless.abstract import Wrapper
from drl.utils.types import Action, Observation


class NoopResetWrapper(Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    """
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            noop_action: Action,
            noop_min: int,
            noop_max: int):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            noop_action (int): No-op action.
            noop_min (int): Minimum number of no-op actions to take.
            noop_max (int): Maximum number of no-op actions to take.
        """
        super().__init__(env)
        self._noop_action = noop_action
        self._noop_min = noop_min
        self._noop_max = noop_max
        self._run_checks()

    def _run_checks(self) -> None:
        meanings = self.env.unwrapped.get_action_meanings()
        cond = meanings[self._noop_action] == 'NOOP'
        if not cond:
            msg = "Chosen no-op action does not have meaning 'NOOP'."
            raise ValueError(msg)

    @property
    def noop_action(self) -> Action:
        return self._noop_action

    @property
    def noop_min(self) -> int:
        return self._noop_min

    @property
    def noop_max(self) -> int:
        return self._noop_max

    def reset(self, **kwargs: Mapping[str, Any]) -> Observation:
        """
        Takes the no-op action `self.noop_action` a random number of times
        between `self.noop_min` and `self.noop_max`, inclusive.
        """
        obs = self.env.reset(**kwargs)
        low, high = self.noop_min, self.noop_max + 1
        noops = self.unwrapped.np_random.randint(low, high)
        if noops == 0:
            return obs
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
