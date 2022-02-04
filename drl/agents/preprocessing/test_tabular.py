import torch as tc

from drl.agents.preprocessing.tabular import one_hot, OneHotEncode

depth = 3


def test_one_hot():
    tc.testing.assert_close(
        actual=tc.tensor([[1, 0, 0]]).float(),
        expected=one_hot(tc.tensor([0]), depth=depth),
        rtol=1e-4,
        atol=1e-4)


def test_one_hot_encode():
    preproc = OneHotEncode(depth=depth)
    tc.testing.assert_close(
        actual=tc.tensor([[1, 0, 0]]).float(),
        expected=preproc(tc.tensor([0])),
        rtol=1e-4,
        atol=1e-4)
