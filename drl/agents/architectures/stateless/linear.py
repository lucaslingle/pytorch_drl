from typing import Tuple, Mapping, Any

import torch as tc

from drl.agents.architectures.stateless.abstract import StatelessArchitecture


class Linear(StatelessArchitecture):
    """
    Linear architecture.
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]]
    ):
        """
        Args:
            input_dim: Input dimensionality.
            output_dim: Output dimensionality.
            w_init_spec: Tuple containing weight initializer name and kwargs.
            b_init_spec: Tuple containing bias initializer name and kwargs.
        """
        super().__init__(w_init_spec, b_init_spec)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._network = tc.nn.Linear(input_dim, output_dim)
        self._init_weights()

    @property
    def input_shape(self):
        shape = (self._input_dim,)
        return shape

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x, **kwargs):
        features = self._network(x)
        return features
