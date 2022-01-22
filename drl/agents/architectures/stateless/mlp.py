from typing import Tuple, Mapping, Any, Optional

import torch as tc

from drl.agents.architectures.stateless.abstract import StatelessArchitecture


class MLP(StatelessArchitecture):
    """
    Multilayer perceptron architecture.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]],
            num_layers: int = 2,
            hidden_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Input dimensionality.
            hidden_dim: Intermediate layer dimensionality.
            output_dim: Output dimensionality.
            w_init_spec: Tuple containing weight initializer name and kwargs.
            b_init_spec: Tuple containing bias initializer name and kwargs.
        """
        super().__init__(w_init_spec, b_init_spec)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim if hidden_dim else 2 * input_dim
        self._output_dim = output_dim
        self._network = tc.nn.Sequential(*[
            tc.nn.Sequential(
                tc.nn.Linear(
                    in_features=input_dim if l == 0 else hidden_dim,
                    out_features=output_dim if l == num_layers-1 else hidden_dim),
                tc.nn.ReLU())
            for l in range(num_layers)
        ])
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
