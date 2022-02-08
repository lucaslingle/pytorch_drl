import torch as tc
import numpy as np

from drl.algos.common.losses import get_loss


def test_mse_loss():
    criterion = get_loss('MSELoss')
    loss = criterion(input=tc.tensor([0.]), target=tc.tensor([1.0])).mean()
    tc.testing.assert_close(actual=loss, expected=tc.tensor(1.0))


def test_smoothl1_loss():
    criterion = get_loss('SmoothL1Loss')
    loss = criterion(input=tc.tensor([0.]), target=tc.tensor([1.0])).mean()
    tc.testing.assert_close(actual=loss, expected=tc.tensor(0.5 * 1.0**2))

    loss = criterion(input=tc.tensor([0.]), target=tc.tensor([0.5])).mean()
    tc.testing.assert_close(actual=loss, expected=tc.tensor(0.5 * 0.5**2))

    loss = criterion(input=tc.tensor([0.]), target=tc.tensor([2.0])).mean()
    tc.testing.assert_close(actual=loss, expected=tc.tensor(2.0 - 0.5))
