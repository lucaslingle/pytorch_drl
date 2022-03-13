from typing import Mapping, Any
from collections import Counter

from drl.utils.types import Observation, Action, EnvOutput
from drl.envs.wrappers.stateless import Wrapper


class CounterWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._counts = Counter()

    @property
    def counts(self):
        return self._counts

    def reset(
            self,
            reset_counts: bool = False,
            **kwargs: Mapping[str, Any]) -> Observation:
        if reset_counts:
            self._counts = Counter()
        return self.env.reset(**kwargs)

    def step(self, action: Action) -> EnvOutput:
        self._counts[action] += 1
        return self.env.step(action)
