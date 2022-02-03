import torch as tc

from drl.agents.architectures.stateless.identity import Identity
from drl.utils.initializers import get_initializer


def test_identity():
    batch_size = 32
    input_dim = 512
    input_shape = [input_dim]
    img_batch = tc.zeros(size=(batch_size, *input_shape), dtype=tc.float32)
    identity = Identity(
        input_shape=input_shape,
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    assert identity.input_shape == input_shape
    assert identity.output_dim == input_dim
    tc.testing.assert_close(
        actual=identity(img_batch),
        expected=img_batch,
        rtol=1e-4,
        atol=1e-4)
