import abc

import torch as tc

from drl.agents.heads.abstract import Head


class CategoricalPolicyHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        logits = self._policy_head(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class DiagonalGaussianPolicyHead(Head, metaclass=abc.ABCMeta):
    def forward(self, features, **kwargs):
        vec = self._policy_head(features)
        mu, logsigma = tc.chunk(vec, 2, dim=-1)
        dist = tc.distributions.Normal(loc=mu, scale=tc.exp(logsigma))
        return dist


class EpsilonGreedyCategoricalPolicyHead(CategoricalPolicyHead):
    def __init__(self, action_value_head, **kwargs):
        super().__init__()
        self._action_value_head = action_value_head

    def forward(self, features, epsilon, **kwargs):
        q_values = self._action_value_head(features)
        probs = tc.tensor([epsilon, 1-epsilon])
        dist = tc.distributions.Categorical(probs=probs)
        do_random = dist.sample().bool().item()
        if do_random:
            return tc.randint(0, high=q_values.shape[-1], size=(1,)).squeeze(-1)
        return tc.argmax(q_values, dim=-1, keepdim=True)


class LinearCategoricalPolicyHead(CategoricalPolicyHead):
    def __init__(self, num_features, num_actions, **kwargs):
        super().__init__()
        self._policy_head = tc.nn.Linear(num_features, num_actions)


class LinearDiagonalGaussianPolicyHead(DiagonalGaussianPolicyHead):
    def __init__(self, num_features, action_dim, **kwargs):
        super().__init__()
        self._policy_head = tc.nn.Linear(num_features, 2*action_dim)
