from typing import Mapping, Any
from collections import Counter

import torch as tc

from drl.envs.wrappers.stateful import TrainableWrapper


class DummyTrainableWrapper(TrainableWrapper):
    """
    Dummy trainable wrapper.

    Useful for testing handling of trainable wrappers by derived classes of `Algo`.
    """
    def __init__(self, env):
        super().__init__(env)
        self._counts = Counter()

    @property
    def counts(self):
        return self._counts

    def learn(
            self, mb: Mapping[str, tc.Tensor], **kwargs: Mapping[str,
                                                                 Any]) -> None:
        mb_size = mb['observations'].shape[0]
        for idx in range(mb_size):
            o_idx = mb['observations'][idx]
            self._counts[str(o_idx.detach().numpy())] += 1
