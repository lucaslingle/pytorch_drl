from typing import Tuple, Mapping, Any, Callable
import importlib
import functools

import torch as tc
import numpy as np


def get_initializer(
        init_spec: Tuple[str, Mapping[str,
                                      Any]]) -> Callable[[tc.Tensor], None]:
    """
    Args:
        init_spec (Tuple[str, Mapping[str, Any]]): Tuple containing initializer
            name, which should be either be 'normc_' or an initializer from
            torch.nn.init, and initializer args, which should be a dictionary of
            arguments for the initializer.

    Returns:
         Callable[[torch.Tensor], None]: Initializer as a partial function.
    """
    name, args = init_spec
    if name == 'normc_':
        return normc_
    if name == 'eye':
        return eye_
    module = importlib.import_module('torch.nn.init')
    initializer = getattr(module, name)
    return functools.partial(initializer, **args)


def normc_(weight_tensor: tc.Tensor, gain: float = 1.0) -> None:
    """
    Initializes each output neuron's weights with a sample from a
    uniform distribution over the unit hypersphere, as in
    https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97

    Args:
        weight_tensor (torch.Tensor): Weight tensor to initialize.
        gain (float): Gain parameter to scale the samples by. Default: 1.0.

    Returns:
        None
    """
    # note that compared to the baselines reference,
    # which uses tensorflow's input-first tensor format,
    # pytorch uses an output-first format,
    # and also stores convolution kernels differently!
    # code below accounts for this.
    num_dims = len(list(weight_tensor.shape))
    normalize_dims = list(range(num_dims))
    normalize_dims.remove(0)
    normalize_dims = tuple(normalize_dims)
    out = np.random.normal(loc=0.0, scale=1.0, size=weight_tensor.shape)
    out /= np.sqrt(np.sum(np.square(out), axis=normalize_dims, keepdims=True))
    out *= gain
    with tc.no_grad():
        weight_tensor.copy_(tc.tensor(out, requires_grad=False))


def eye_(weight_tensor: tc.Tensor) -> None:
    """
    Initializes the j-th output neuron's weights to e_j,
    the j-th standard basis vector. Only works with 2D square weights.

    Args:
        weight_tensor (torch.Tensor): Weight tensor to initialize.

    Returns:
        None
    """
    num_dims = len(list(weight_tensor.shape))
    assert num_dims == 2
    assert weight_tensor.shape[0] == weight_tensor.shape[1]
    output_dim = weight_tensor.shape[0]
    eye = np.eye(output_dim)
    with tc.no_grad():
        weight_tensor.copy_(tc.tensor(eye, requires_grad=False))
