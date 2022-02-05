"""
Abstract wrapper definitions.
"""

from typing import Union, Mapping, Any, Tuple, Optional, Dict
import abc

import gym

from drl.utils.typing import Checkpointable, Action, Observation, Reward, EnvOutput


class RewardSpec:
    """
    Specifies the reward keys for an environment.

    We use reward keys to distinguish between raw and processed rewards,
    as well as between intrinsic and extrinsic ones.
    """
    def __init__(self, keys):
        self._keys = keys

    @property
    def keys(self):
        return self._keys


class Wrapper(metaclass=abc.ABCMeta):
    """
    Environment wrapper class with built-in support for RewardSpec objects.
    """
    def __init__(self, env: Union[gym.core.Env, 'Wrapper']):
        self.env = env
        self._observation_space = None
        self._action_space = None
        self._reward_spec = None
        self._metadata = None

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Observation space getter.

        Returns:
            gym.core.Space: Observation space.
        """
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: gym.spaces.Space) -> None:
        """
        Observation space setter.

        Args:
            space (gym.spaces.Space): Observation space.

        Returns:
            None.
        """
        self._observation_space = space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Action space getter.

        Returns:
            gym.spaces.Space: Action space.
        """
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: gym.spaces.Space) -> None:
        """
        Action space setter.

        Args:
            space (gym.spaces.Space): Action space.

        Returns:
            None.
        """
        self._action_space = space

    @property
    def reward_spec(self) -> Optional[RewardSpec]:
        """
        Reward spec getter.

        Returns:
            RewardSpec: Reward spec.
        """
        if self._reward_spec is None:
            if isinstance(self.env, gym.core.Env):
                return None
            else:
                return self.env.reward_spec
        return self._reward_spec

    @reward_spec.setter
    def reward_spec(self, spec: RewardSpec) -> None:
        """
        Reward spec setter.

        Args:
            spec (RewardSpec): Reward spec.

        Returns:
            None.
        """
        self._reward_spec = spec

    @property
    def metadata(self):
        """
        Metadata getter.

        Returns:
            Mapping[str, Any]: Metadata.
        """
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value: Mapping[str, Any]):
        """
        Metadata setter.

        Args:
            value (Mapping[str, Any]): Metadata.

        Returns:
            None.
        """
        self._metadata = value

    def step(self, action: Action) -> EnvOutput:
        """
        Steps the wrapped version of the environment.

        Args:
            action (Action): Action to take.

        Returns:
            EnvStepOutput:
                Tuple containing next observation, current reward(s),
                indicator if next state is terminal, and info dictionary.
        """
        return self.env.step(action)

    def reset(self, **kwargs: Mapping[str, Any]) -> Observation:
        """
        Resets the wrapped version of the environment. The RNG state is not reset.

        Args:
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            Observation: Initial observation.
        """
        return self.env.reset(**kwargs)

    def render(self,
               mode: str = "human",
               **kwargs: Mapping[str, Any]) -> Union[Observation, bool]:
        """
        Renders the environment.

        Args:
            mode (str): Mode for rendering. Typically one of 'rgb_array', 'human'.
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            Union[Observation, bool]: RGB array or bool indicating
                rendering success.
        """
        return self.env.render(mode, **kwargs)

    def close(self) -> None:
        """
        Closes the environment.

        Returns:
            None.
        """
        return self.env.close()

    def seed(self, seed: Optional[int] = None) -> Tuple[int, int]:
        """
        Sets the seed of the random number generator in this environment.

        Args:
            seed (Optional[int]): Environment RNG seed.

        Returns:
            Tuple[int, int]: RNG details.
        """
        return self.env.seed(seed)

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self) -> gym.core.Env:
        """
        Unwraps the wrapper, returning the innermost environment.

        Returns:
            gym.core.Env: OpenAI gym environment.
        """
        return self.env.unwrapped

    @property
    def checkpointables(self) -> Dict[str, Checkpointable]:
        """
        Get checkpointable components from all wrappers so far.
        If there are none, returns empty dictionary.

        Returns:
            Dict[str, Checkpointable]: Dictionary of checkpointables.
        """
        if isinstance(self.env, gym.core.Env):
            return dict()
        elif isinstance(self.env, Wrapper):
            return self.env.checkpointables
        else:
            raise NotImplementedError


class ObservationWrapper(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def observation(self, observation: Observation) -> Observation:
        """
        Abstract method for observation transformation function.

        Args:
            observation (Observation): Observation.

        Returns:
            Observation: Transformed observation.
        """
        raise NotImplementedError

    def reset(self, **kwargs: Mapping[str, Any]) -> Observation:
        """
        Resets the wrapped version of the environment.
        The RNG state is not reset.

        Args:
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            Observation: Transformed initial observation.
        """
        return self.observation(self.env.reset(**kwargs))

    def step(self, action: Action) -> EnvOutput:
        """
        Steps the wrapped version of the environment.

        Args:
            action (Action): Action to take.

        Returns:
            EnvOutput: Tuple containing transformed next observation,
                current reward(s), indicator if next state is terminal,
                and info dictionary.
        """
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info


class ActionWrapper(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def action(self, action: Action) -> Action:
        """
        Abstract method for action transformation function.

        Args:
            action (Action): Action to take.

        Returns:
            Action: Transformed action.
        """
        raise NotImplementedError

    def step(self, action: Action) -> EnvOutput:
        """
        Steps the wrapped version of the environment, by transforming the action.

        Args:
            action (Action): Action to take.

        Returns:
            EnvOutput: Tuple containing next observation, current reward(s),
                indicator if next state is terminal, and info dictionary.
        """
        return self.env.step(self.action(action))


class RewardWrapper(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reward(self, reward: Reward) -> Reward:
        """
        Abstract method for reward transformation function.

        Args:
            reward (Reward): A reward to be transformed.

        Returns:
            Reward: Transformed reward.
        """
        raise NotImplementedError

    def step(self, action: Action) -> EnvOutput:
        """
        Steps the wrapped version of the environment.

        Args:
            action (Action): Action.

        Returns:
            EnvOutput: Tuple containing next observation, transformed
                current reward(s), indicator if next state is terminal,
                and info dictionary.
        """
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info
