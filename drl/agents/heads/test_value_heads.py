import torch as tc

from drl.agents.heads.value_heads import SimpleValueHead
from drl.agents.architectures import Linear
from drl.utils.initializers import get_initializer

batch_size = 1
num_features = 10
num_actions = 4


def test_simple_value_head():
    head = SimpleValueHead(
        num_features=num_features,
        head_architecture_cls=Linear,
        head_architecture_cls_args={},
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    features_batch = tc.zeros(size=[batch_size, num_features], dtype=tc.float32)
    vpred = head(features_batch)
    tc.testing.assert_close(
        actual=vpred,
        expected=tc.zeros(size=[batch_size], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4)
