import torch as tc

from drl.agents.heads.action_value_heads import (
    SimpleDiscreteActionValueHead,
    SimpleContinuousActionValueHead,
    DistributionalDiscreteActionValueHead)
from drl.agents.architectures import Linear
from drl.utils.initializers import get_initializer


batch_size = 8
num_features = 10
num_actions = 4


def test_simple_discrete_action_value_head():
    head = SimpleDiscreteActionValueHead(
        num_features=num_features,
        num_actions=num_actions,
        head_architecture_cls=Linear,
        head_architecture_cls_args={},
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    features_batch = tc.zeros(size=[batch_size, num_features], dtype=tc.float32)
    q_preds = head(features_batch)
    tc.testing.assert_close(
        actual=q_preds,
        expected=tc.zeros(size=[batch_size, num_actions], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4)


def test_simple_continuous_action_value_head():
    head = SimpleContinuousActionValueHead(
        num_features=num_features,
        head_architecture_cls=Linear,
        head_architecture_cls_args={},
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    features_batch = tc.zeros(size=[batch_size, num_features], dtype=tc.float32)
    q_pred = head(features_batch)
    tc.testing.assert_close(
        actual=q_pred,
        expected=tc.zeros(size=[batch_size], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4)


def test_distributional_discrete_action_value_head():
    vmin = -10.
    vmax = 10.
    num_bins = 51
    head = DistributionalDiscreteActionValueHead(
        num_features=num_features,
        num_actions=num_actions,
        head_architecture_cls=Linear,
        head_architecture_cls_args={},
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})),
        vmin=vmin,
        vmax=vmax,
        num_bins=num_bins)
    assert head.returns_to_bin_ids(tc.tensor([1000.0])) == tc.tensor([51])
    features_batch = tc.zeros(size=[batch_size, num_features], dtype=tc.float32)
    q_value_logits = head(features_batch)
    tc.testing.assert_close(
        actual=q_value_logits,
        expected=tc.zeros(
            size=[batch_size, num_actions, num_bins], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4)

    # logits should be uniform with head weights all zero,
    # and with symmetric return bins about 0.0,
    # the average value of each distributional prediction should be zerp.
    tc.testing.assert_close(
        actual=head.logits_to_mean(q_value_logits),
        expected=tc.zeros(size=[batch_size, num_actions], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4)
