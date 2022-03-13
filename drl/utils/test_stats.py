import torch as tc

from drl.utils.stats import standardize


def make_tensor():
    return tc.arange(10).float()


def test_standardize():
    tensor = make_tensor()
    standardized = standardize(tensor)
    tc.testing.assert_close(actual=standardized.mean(), expected=tc.tensor(0.))
    tc.testing.assert_close(actual=standardized.std(), expected=tc.tensor(1.))
