import torch as tc

from drl.agents.architectures.stateless.dueling import DuelingArchitecture
from drl.utils.initializers import get_initializer


def test_dueling():
    batch_size = 32
    input_dim, num_actions = 512, 18
    input_shape = [input_dim]
    img_batch = tc.zeros(size=(batch_size, *input_shape), dtype=tc.float32)
    dueling = DuelingArchitecture(
        input_dim=input_dim,
        output_dim=num_actions,
        widening=1,
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    assert dueling.input_shape == input_shape
    assert dueling.output_dim == num_actions
    tc.testing.assert_close(
        actual=dueling(img_batch),
        expected=tc.zeros(size=[batch_size, num_actions], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4)
