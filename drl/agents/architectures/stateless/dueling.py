from typing import Callable

import torch as tc

from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture
from drl.agents.architectures.stateless.mlp import MLP


class DuelingArchitecture(HeadEligibleArchitecture):
    """
    Reference:
        Z. Wang et al., 2016 -
            'Dueling Network Architectures for Deep Reinforcement Learning'
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            widening: int,
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None]):
        """
        Args:
            input_dim (int): Input dimensionality.
            output_dim (int): Output dimensionality (number of actions).
            widening (int): Widening factor for multiplying the number of
                internal features present in the original architecture
                of Wang et al., 2016.
            w_init (Callable[[torch.Tensor], None]): Weight initializer.
            b_init (Callable[[torch.Tensor], None]): Bias initializer.
        """
        super().__init__(input_dim, output_dim, w_init, b_init)
        self._proj = tc.nn.Sequential(
            tc.nn.Linear(input_dim, 50 * widening), tc.nn.ReLU())
        self._advantage_stream = MLP(
            input_dim=50 * widening,
            hidden_dim=25 * widening,
            output_dim=output_dim,
            num_layers=2,
            w_init=w_init,
            b_init=b_init)
        self._value_stream = MLP(
            input_dim=50 * widening,
            hidden_dim=25 * widening,
            output_dim=1,
            num_layers=2,
            w_init=w_init,
            b_init=b_init)
        self._init_weights(self._proj)

    def forward(self, features, **kwargs):
        proj_features = self._proj(features)
        adv_stream = self._advantage_stream(proj_features)
        adv_stream -= adv_stream.mean(dim=-1, keepdim=True)
        val_stream = self._value_stream(proj_features)
        qpreds = adv_stream + val_stream
        return qpreds
