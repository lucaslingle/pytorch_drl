from typing import Callable, List

import torch as tc
import numpy as np

from drl.agents.architectures.stateless.abstract import StatelessArchitecture


class Identity(StatelessArchitecture):
    """
    Identity architecture. Useful for unit testing.
    """
    def __init__(
            self,
            input_shape: List[int],
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None]
    ):
        """
        Args:
            input_dim (List[int]): Input shape.
            w_init (Callable[[tc.Tensor], None]): Weight initializer.
            b_init (Callable[[tc.Tensor], None]): Bias initializer.
        """
        super().__init__(w_init, b_init)
        self._input_shape = input_shape

    @property
    def input_shape(self) -> List[int]:
        return self._input_shape

    @property
    def output_dim(self) -> int:
        return np.prod(self._input_shape)

    def forward(self, x, **kwargs):
        assert list(x.shape[1:]) == self.input_shape
        features = x.reshape(-1, self.output_dim)
        return features
