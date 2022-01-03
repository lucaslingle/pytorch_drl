from drl.agents.heads.abstract import Head
from drl.agents.heads.policy_heads import (
    CategoricalPolicyHead,
    DiagonalGaussianPolicyHead,
    LinearCategoricalPolicyHead,
    LinearDiagonalGaussianPolicyHead
)
from drl.agents.heads.value_heads import (
    ValueHead,
    LinearValueHead
)


__all__ = [
    "Head",
    "CategoricalPolicyHead",
    "DiagonalGaussianPolicyHead",
    "LinearCategoricalPolicyHead",
    "LinearDiagonalGaussianPolicyHead",
    "LinearValueHead"
]
