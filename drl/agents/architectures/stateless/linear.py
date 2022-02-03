from typing import Callable

import torch as tc

from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture


class Linear(HeadEligibleArchitecture):
    """
    Linear architecture.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
    ):
        """
        Args:
            input_dim (int): Input dimensionality.
            output_dim (int): Output dimensionality.
            w_init (Callable[[torch.Tensor], None]): Weight initializer.
            b_init (Callable[[torch.Tensor], None]): Bias initializer.
        """
        super().__init__(input_dim, output_dim, w_init, b_init)
        self._network = tc.nn.Sequential(
            tc.nn.Linear(input_dim, output_dim)
        )
        self._init_weights(self._network)

    def forward(self, x, **kwargs):
        features = self._network(x)
        return features
