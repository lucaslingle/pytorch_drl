import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.common import normc_init_


class CategoricalActionValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        q_values = self._action_value_head(features)
        return q_values


class ContinuousActionValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        q_value = self._action_value_head(features).squeeze(-1)
        return q_value


class LinearCategoricalActionValueHead(CategoricalActionValueHead):
    def __init__(self, num_features, num_actions, ortho_init, **kwargs):
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, num_actions)
        self._ortho_init = ortho_init
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(self._action_value_head.weight, gain=0.01)
        else:
            normc_init_(self._action_value_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._value_head.bias)


class LinearContinuousActionValueHead(ContinuousActionValueHead):
    def __init__(self, num_features, ortho_init, **kwargs):
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, 1)
        self._ortho_init = ortho_init
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(self._action_value_head.weight, gain=0.01)
        else:
            normc_init_(self._action_value_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._value_head.bias)
