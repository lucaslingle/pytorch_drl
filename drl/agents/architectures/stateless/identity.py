from typing import Callable, Mapping, Any

import torch as tc
import numpy as np

from drl.agents.architectures.stateless.abstract import StatelessArchitecture


class Identity(StatelessArchitecture):
    """
    Identity architecture. Useful for unit testing.
    """
    def __init__(
            self,
            input_shape: int,
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            input_dim: Input shape.
            w_init: Weight initializer.
            b_init: Bias initializer.
            **kwargs: Keyword arguments.
        """
        super().__init__(w_init, b_init)
        self._input_shape = input_shape

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_dim(self):
        return np.prod(self._input_shape)

    def forward(self, x, **kwargs):
        assert x.shape[1:] == self.input_shape
        features = x.reshape(-1, self.output_dim)
        return features
