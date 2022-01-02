"""
Abstract wrapper definitions.
"""

from typing import Optional, Union
import abc

import gym


# todo(lucaslingle):
#   figure out a good way to establish a partial ordering over subtypes of
#   Union[gym.core.Env, 'Wrapper']...
#   Use it to prevent wrappers from being ordered impermissably.

class Wrapper(metaclass=abc.ABCMeta):
    def __init__(self, env: Union[gym.core.Env, 'Wrapper']):
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def action_space(self):
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self._action_space = space

    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space):
        self._observation_space = space

    @property
    def metadata(self):
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed: Optional[int] = None, **kwargs):
        return self.env.reset(seed, **kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def get_checkpointables(self):
        if isinstance(self.env, gym.core.Env):
            return dict()
        elif isinstance(self.env, Wrapper):
            return self.env.get_checkpointables()
        else:
            raise NotImplementedError


class ObservationWrapper(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def observation(self, observation):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, **kwargs):
        return self.observation(self.env.reset(seed=seed, **kwargs))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info


class RewardWrapper(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reward(self, reward):
        raise NotImplementedError

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info


class ActionWrapper(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def action(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def reverse_action(self, action):
        raise NotImplementedError

    def step(self, action):
        return self.env.step(self.action(action))
