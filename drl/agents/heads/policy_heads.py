import torch as tc

from drl.agents.heads.abstract import (
    CategoricalPolicyHeadMixin,
    DiagonalGaussianPolicyHeadMixin,
)


class LinearCategoricalPolicyHead(CategoricalPolicyHeadMixin):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self.__policy_head = tc.nn.Linear(num_features, num_actions)


class LinearDiagonalGaussianPolicyHead(DiagonalGaussianPolicyHeadMixin):
    def __init__(self, num_features, action_dim):
        super().__init__()
        self.__policy_head = tc.nn.Linear(num_features, 2*action_dim)
