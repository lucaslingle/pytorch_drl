import torch as tc
import numpy as np


class GroupNorm(tc.nn.Module):
    def __init__(self, num_filters, group_size):
        assert num_filters % group_size == 0
        super().__init__()
        self._num_filters = num_filters
        self._group_size = group_size
        self._num_groups = num_filters // group_size
        self._g = tc.nn.Parameter(
            tc.ones((self._num_groups, self._group_size), dtype=tc.float32))
        self._b = tc.nn.Parameter(
            tc.zeros((self._num_groups, self._group_size), dtype=tc.float32))

    def forward(self, x):
        h, w = x.shape[-2:]
        grouped = x.reshape(-1, self._num_groups, self._group_size, h, w)
        mean = grouped.mean(dim=(-3,-2,-1), keepdims=True)
        zeromean = grouped - mean
        var = zeromean.square().mean(dim=(-3,-2,-1), keepdims=True)
        normalized = zeromean * tc.rsqrt(var + 1e-6)
        g = self._g.reshape(1, *self._g.shape, 1, 1)
        b = self._b.reshape(1, *self._b.shape, 1, 1)
        affine = g * normalized + b
        return affine.reshape(x.shape)


class MaybeGroupNorm(tc.nn.Module):
    def __init__(self, num_filters, num_groups, use_gn):
        super().__init__()
        self._use_gn = use_gn
        if self._use_gn:
            self._gn = GroupNorm(num_filters, num_groups)

    def forward(self, x):
        if self._use_gn:
            return self._gn(x)
        return x


def normc_init_(weight_tensor, gain=1.0):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97
    Note that in tensorflow the weight tensor in a linear layer is stored with the
    input dim first and the output dim second. See
    https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/layers/core.py#L1193

    In contrast, in pytorch the output dim is first. See
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

    This means if we want a normc init in pytorch,
    we have to change which dim(s) we normalize on.
    In addition, for convolutions in pytorch, the height and width are last.
    """
    num_dims = len(list(weight_tensor.shape))
    normalize_dims = list(range(num_dims))
    normalize_dims.remove(0)
    normalize_dims = tuple(normalize_dims)
    out = np.random.normal(loc=0.0, scale=1.0, size=weight_tensor.shape)
    out /= np.sqrt(np.sum(np.square(out), axis=normalize_dims, keepdims=True))
    out *= gain
    with tc.no_grad():
        weight_tensor.copy_(tc.tensor(out, requires_grad=False))
