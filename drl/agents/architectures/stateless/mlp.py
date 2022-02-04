from typing import Callable

import torch as tc

from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture


class MLP(HeadEligibleArchitecture):
    """
    Multilayer perceptron architecture.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        w_init: Callable[[tc.Tensor], None],
        b_init: Callable[[tc.Tensor], None],
    ):
        """
        Args:
            input_dim (int): Input dimensionality.
            hidden_dim (int): Intermediate layer dimensionality.
            output_dim (int): Output dimensionality.
            num_layers (int): Number of layers.
            w_init (Callable[[torch.Tensor], None]): Weight initializer.
            b_init (Callable[[torch.Tensor], None]): Bias initializer.
        """
        super().__init__(input_dim, output_dim, w_init, b_init)
        if not num_layers > 1:
            raise ValueError("MLP requires num_layers > 1.")
        self._network = tc.nn.Sequential(
            *[
                tc.nn.Linear(
                    in_features=input_dim if l // 2 == 0 else hidden_dim,
                    out_features=output_dim if l // 2 == num_layers -
                    1 else hidden_dim) if l % 2 == 0 else tc.nn.ReLU()
                for l in range(2 * num_layers)
            ])
        self._init_weights(self._network)

    def forward(self, x, **kwargs):
        features = self._network(x)
        return features
