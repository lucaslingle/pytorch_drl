import torch as tc

from drl.agents.architectures.stateless.linear import Linear
from drl.utils.initializers import get_initializer


def test_linear():
    batch_size = 32
    input_dim, output_dim = 512, 18
    input_shape = [input_dim]
    img_batch = tc.zeros(size=(batch_size, *input_shape), dtype=tc.float32)
    linear = Linear(
        input_dim=input_dim,
        output_dim=output_dim,
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    assert linear.input_shape == input_shape
    assert linear.output_dim == output_dim
    tc.testing.assert_close(
        actual=linear(img_batch),
        expected=tc.zeros(size=[batch_size, output_dim], dtype=tc.float32))
