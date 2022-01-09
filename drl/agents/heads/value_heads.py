import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.common import normc_init_


class ValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        vpred = self._value_head(features).squeeze(-1)
        return vpred


class LinearValueHead(ValueHead):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self._value_head = tc.nn.Linear(num_features, 1)
        self._init_weights()

    def _init_weights(self):
        normc_init_(self._value_head.weight, gain=1.0)

