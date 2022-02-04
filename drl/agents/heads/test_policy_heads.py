import torch as tc

from drl.agents.heads.policy_heads import CategoricalPolicyHead, DiagonalGaussianPolicyHead
from drl.agents.architectures import Linear
from drl.utils.initializers import get_initializer

num_features = 10
num_actions = 4


def test_categorical_policy_head():
    head = CategoricalPolicyHead(
        num_features=num_features,
        num_actions=num_actions,
        head_architecture_cls=Linear,
        head_architecture_cls_args={},
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    features_batch = tc.zeros(size=[1, num_features], dtype=tc.float32)
    dist = head(features_batch)
    tc.testing.assert_close(
        actual=dist.log_prob(tc.tensor([0])),
        expected=tc.log(tc.tensor([1. / num_actions])),
        rtol=1e-4,
        atol=1e-4)


def test_diagonal_gaussian_policy_head():
    head = DiagonalGaussianPolicyHead(
        num_features=num_features,
        action_dim=num_actions,
        head_architecture_cls=Linear,
        head_architecture_cls_args={},
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    features_batch = tc.zeros(size=[1, num_features], dtype=tc.float32)
    actions_batch = tc.zeros(size=[1, num_actions], dtype=tc.float32)
    logprobs_batch = -0.5 * num_actions * tc.log(tc.tensor([2.0 * tc.pi]))
    dist = head(features_batch)
    tc.testing.assert_close(
        actual=dist.log_prob(actions_batch),
        expected=logprobs_batch,
        rtol=1e-4,
        atol=1e-4)
