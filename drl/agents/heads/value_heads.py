import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.common import normc_init_


class ValueHead(Head, metaclass=abc.ABCMeta):
    """
    Value head abstract class.
    """


class SimpleValueHead(ValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for simple value prediction heads
    (as opposed to distributional).
    """
    def forward(self, features, **kwargs):
        vpred = self._value_head(features).squeeze(-1)
        return vpred


class DistributionalValueHead(ValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for distributional value prediction heads.
    Reference: Bellemare et al., 2017.
    """
    def forward(self, features, **kwargs):
        vpred = self._value_head(features)
        return vpred


class LinearSimpleValueHead(SimpleValueHead):
    """
    Simple value prediction head using a linear projection of features.
    """
    def __init__(self, num_features, ortho_init, ortho_gain=0.01, **kwargs):
        """
        Args:
            num_features: Number of features.
            ortho_init: Use orthogonal initialization?
            ortho_gain: Gain parameter for orthogonal initialization.
                Ignored if ortho_init is False.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._value_head = tc.nn.Linear(num_features, 1)
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._value_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._value_head.weight, gain=1.0)
        tc.nn.init.zeros_(self._value_head.bias)
