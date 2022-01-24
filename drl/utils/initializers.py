from typing import Tuple, Mapping, Any, Callable
import importlib
import functools

import torch as tc
import numpy as np


def get_initializer(
        init_spec: Tuple[str, Mapping[str, Any]]
) -> Callable[[tc.Tensor], None]:
    """
    Args:
        init_spec: Tuple containing initializer name, which should be either
            'normc_' or an initializer from torch.nn.init, and initializer args,
            which should be a dictionary of arguments for the initializer.

    Returns:
        Initializer as a partial function.
    """
    name, args = init_spec
    if name == 'normc_':
        return normc_
    module = importlib.import_module('torch.nn.init')
    initializer = getattr(module, name)
    return functools.partial(initializer, **args)


def normc_(weight_tensor: tc.Tensor, gain: float = 1.0) -> None:
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
