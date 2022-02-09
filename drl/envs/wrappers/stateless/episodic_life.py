"""
Life-as-episode wrapper.
"""

from typing import Union, Callable, Mapping, Any

import gym

from drl.envs.wrappers.stateless.abstract import Wrapper
from drl.utils.types import Action, Observation, EnvOutput


class EpisodicLifeWrapper(Wrapper):
    """
    Use loss-of-life to mark end-of-episode using the done indicator.
    """
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            lives_fn: Callable[[Union[gym.core.Env, Wrapper]], int],
            noop_action: Action):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            lives_fn (Callable[[Union[gym.core.Env, Wrapper]], int]): Function
                to obtain number of lives from the provided env.
            noop_action (ActionType): Any no-op action for the provided env.
        """
        super().__init__(env)
        self._was_real_done = True
        self._lives = 0
        self._lives_fn = lives_fn
        self._noop_action = noop_action

    def step(self, action: Action) -> EnvOutput:
        obs, reward, self._was_real_done, info = self.env.step(action)
        lives = self._lives_fn(self.env)
        done = self._was_real_done or (0 < lives < self._lives)
        self._lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs: Mapping[str, Any]) -> Observation:
        """
        Resets only when lives are exhausted.
        """
        if self._was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, *_ = self.env.step(self._noop_action)
        self._lives = self._lives_fn(self.env)
        return obs
