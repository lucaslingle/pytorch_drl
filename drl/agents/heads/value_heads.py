import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.common import normc_init_


class ValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        vpred = self._value_head(features).squeeze(-1)
        return vpred


class LinearValueHead(ValueHead):
    def __init__(self, num_features, ortho_init, **kwargs):
        super().__init__()
        self._value_head = tc.nn.Linear(num_features, 1)
        self._ortho_init = ortho_init
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(self._value_head.weight, gain=1.0)
        else:
            normc_init_(self._value_head.weight, gain=1.0)
        tc.nn.init.zeros_(self._value_head.bias)
