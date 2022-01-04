import abc

import torch as tc

from drl.agents.heads.abstract import Head


class CategoricalActionValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        q_values = self._action_value_head(features)
        return q_values


class ContinuousActionValueHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        q_value = self._action_value_head(features).squeeze(-1)
        return q_value


class LinearCategoricalActionValueHead(CategoricalActionValueHead):
    def __init__(self, num_features, num_actions, **kwargs):
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, num_actions)


class LinearContinuousActionValueHead(ContinuousActionValueHead):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, 1)
