"""
Time limit wrapper.
"""

from typing import Union, Mapping, Any

import gym

from drl.envs.wrappers.stateless.abstract import Wrapper
from drl.utils.typing import Action, Observation, EnvOutput


class TimeLimitWrapper(Wrapper):
    def __init__(
            self, env: Union[gym.core.Env, Wrapper], max_episode_steps: int):
        """
        Args:
            env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
            max_episode_steps (int): Maximum number of env steps.
        """
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: Action) -> EnvOutput:
        if self._elapsed_steps is None:
            msg = "Cannot call env.step() before calling reset()"
            raise RuntimeError(msg)
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs: Mapping[str, Any]) -> Observation:
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
