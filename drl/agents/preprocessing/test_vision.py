import torch as tc

from drl.agents.preprocessing.vision import ToChannelMajor


def test_to_channel_major():
    preproc = ToChannelMajor()
    x = tc.arange(8).reshape(1, 2, 2, 2)
    y = tc.tensor([[[[0, 2], [4, 6]], [[1, 3], [5, 7]]]])
    tc.testing.assert_close(actual=preproc(x), expected=y, rtol=1e-4, atol=1e-4)
