import pytest
import torch as tc

from drl.agents.architectures.stateless.mlp import MLP
from drl.utils.initializers import get_initializer

batch_size = 32
input_dim, hidden_dim, output_dim = 512, 42, 18
input_shape = [input_dim]
img_batch = tc.zeros(size=(batch_size, *input_shape), dtype=tc.float32)


def test_mlp_1layer():
    with pytest.raises(ValueError):
        # raise valuerror on num_layers == 1 due to ambiguity about output dim
        # i.e., it is hidden_dim or output_dim?
        mlp1 = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=1,
            w_init=get_initializer(('zeros_', {})),
            b_init=get_initializer(('zeros_', {})))


def test_mlp_2layer():
    mlp2 = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    assert mlp2.input_shape == input_shape
    assert mlp2.output_dim == output_dim
    tc.testing.assert_close(
        actual=mlp2(img_batch),
        expected=tc.zeros(size=[batch_size, output_dim], dtype=tc.float32))


def test_mlp_3layer():
    mlp3 = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    assert mlp3.input_shape == input_shape
    assert mlp3.output_dim == output_dim
    tc.testing.assert_close(
        actual=mlp3(img_batch),
        expected=tc.zeros(size=[batch_size, output_dim], dtype=tc.float32))
