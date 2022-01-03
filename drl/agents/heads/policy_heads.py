import abc

import torch as tc

from drl.agents.heads.abstract import Head


class CategoricalPolicyHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features):
        logits = self._policy_head(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class DiagonalGaussianPolicyHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features):
        vec = self._policy_head(features)
        mu, logsigma = tc.chunk(vec, 2, dim=-1)
        dist = tc.distributions.Normal(loc=mu, scale=tc.exp(logsigma))
        return dist


class LinearCategoricalPolicyHead(CategoricalPolicyHead):
    def __init__(self, num_features, num_actions, **kwargs):
        super().__init__()
        self._policy_head = tc.nn.Linear(num_features, num_actions)


class LinearDiagonalGaussianPolicyHead(DiagonalGaussianPolicyHead):
    def __init__(self, num_features, action_dim, **kwargs):
        super().__init__()
        self._policy_head = tc.nn.Linear(num_features, 2*action_dim)
