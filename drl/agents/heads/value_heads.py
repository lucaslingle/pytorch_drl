import abc

import torch as tc

from drl.agents.heads.abstract import Head


class ValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        vpred = self._value_head(features).squeeze(-1)
        return vpred


class LinearValueHead(ValueHead):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self._value_head = tc.nn.Linear(num_features, 1)
