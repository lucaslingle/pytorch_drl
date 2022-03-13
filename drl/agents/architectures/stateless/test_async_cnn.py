import torch as tc

from drl.agents.architectures.stateless.async_cnn import AsyncCNN
from drl.utils.initializers import get_initializer


def test_async_cnn():
    img_channels = 4
    batch_size = 32
    img_height, img_width = 84, 84
    img_shape = [img_channels, img_height, img_width]
    img_batch = tc.zeros(size=(batch_size, *img_shape), dtype=tc.float32)
    async_cnn = AsyncCNN(
        img_channels=img_channels,
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {})))
    feature_dim = 256
    assert async_cnn.input_shape == img_shape
    assert async_cnn.output_dim == feature_dim
    tc.testing.assert_close(
        actual=async_cnn(img_batch),
        expected=tc.zeros(size=[batch_size, feature_dim], dtype=tc.float32))
