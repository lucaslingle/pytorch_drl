import torch as tc

from drl.utils.initializers import get_initializer, normc_


def test_get_initializer():
    model = tc.nn.Linear(10, 3)

    get_initializer(('zeros_', {}))(model.weight)
    tc.testing.assert_close(
        actual=model.weight, expected=tc.zeros(size=(3, 10), dtype=tc.float32))
    get_initializer(('zeros_', {}))(model.bias)
    tc.testing.assert_close(
        actual=model.bias, expected=tc.zeros(size=(3, ), dtype=tc.float32))

    get_initializer(('ones_', {}))(model.weight)
    tc.testing.assert_close(
        actual=model.weight, expected=tc.ones(size=(3, 10), dtype=tc.float32))
    get_initializer(('ones_', {}))(model.bias)
    tc.testing.assert_close(
        actual=model.bias, expected=tc.ones(size=(3, ), dtype=tc.float32))


def test_normc_():
    lin = tc.nn.Linear(10, 3)
    normc_(lin.weight, gain=1.0)
    for i in range(3):
        tc.testing.assert_close(
            actual=tc.sum(tc.square(lin.weight[i, :])), expected=tc.tensor(1.0))
