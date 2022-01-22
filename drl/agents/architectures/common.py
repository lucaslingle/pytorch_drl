import importlib

import torch as tc
import numpy as np


def get_initializer(name):
    if name == 'normc':
        return normc_init_
    module = importlib.import_module('torch.nn.init')
    initializer = getattr(module, name)
    return initializer


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
