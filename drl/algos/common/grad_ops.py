from typing import Mapping

import numpy as np
import torch as tc

from drl.utils.types import Module, Optimizer, FlatGrad


def _norm(gradient: FlatGrad) -> float:
    return np.sqrt(np.sum(np.square(gradient)) + 1e-6)


@tc.no_grad()
def read_gradient(network: Module, normalize: bool = False) -> FlatGrad:
    """
    Reads the currently stored gradient in the network parameters' grad
    attributes, and returns it as a vector.

    Args:
        network (Module): Torch Module instance.
        normalize (bool): Normalize the gradient extracted? Default: False.

    Returns:
        numpy.ndarray: Gradient as a one-dimensional numpy ndarray.
    """
    gradient_subvecs = []
    for p in network.parameters():
        if p.grad is not None:
            subvec = p.grad.reshape(-1).detach().numpy()
            gradient_subvecs.append(subvec)
    gradient = np.concatenate(gradient_subvecs, axis=0)
    if normalize:
        gradient /= _norm(gradient)
    return gradient


@tc.no_grad()
def write_gradient(network: Module, gradient: FlatGrad) -> None:
    """
    Writes a gradient vector into a network's parameters' grad attributes.

    Args:
        network (torch.nn.Module): Torch Module instance.
        gradient (numpy.ndarray): Gradient as a one-dimensional numpy ndarray.

    Returns:
        None.
    """
    dims_so_far = 0
    for p in network.parameters():
        if p.grad is not None:
            numel = np.prod(p.shape)
            subvec = gradient[dims_so_far:dims_so_far + numel]
            p.grad.copy_(
                tc.tensor(subvec, requires_grad=False).reshape(p.shape))
            dims_so_far += numel


def task_losses_to_grads(
        network: Module,
        optimizer: Optimizer,
        task_losses: Mapping[str, tc.Tensor],
        normalize: bool = False) -> Mapping[str, FlatGrad]:
    """
    Computes a dictionary of task gradients from task losses.

    Args:
        network (torch.nn.Module): Torch Module instance.
        optimizer (torch.optim.Optimizer): Torch Optimizer instance.
        task_losses (Mapping[str, torch.Tensor]): Dictionary mapping from
            task names to losses.
        normalize (bool): Normalize the task gradients? Default: False.

    Returns:
        Mapping[str, FlatGrad]: Dictionary mapping from task names to gradients.
    """
    task_gradients = dict()
    for k in task_losses:
        optimizer.zero_grad()
        task_losses[k].backward(retain_graph=True)
        task_gradients[k] = read_gradient(network, normalize)
    optimizer.zero_grad()
    return task_gradients


def pcgrad_gradient_surgery(task_gradients: Mapping[str, FlatGrad]) -> FlatGrad:
    """
    Implements the PCGrad gradient surgery algorithm.

    Reference:
        T. Yu et al., 2020 -
            'Gradient Surgery for Multi-Task Learning'

    Args:
        task_gradients (Mapping[str, np.ndarray]): Dictionary mapping from
            task names to gradients.

    Returns:
        numpy.ndarray: PCGrad gradient as a vector.
    """
    pcgrad_gradients = list()
    for i in task_gradients:
        grad_i = task_gradients[i]
        pcgrad_i = grad_i
        for j in task_gradients:
            if i == j:
                continue
            grad_j = task_gradients[j]
            if np.dot(pcgrad_i, grad_j) < 0.:
                coef = np.dot(pcgrad_i, grad_j) / np.square(_norm(grad_j))
                pcgrad_i = pcgrad_i - coef * grad_j
                # bug caught: dont do inplace subtraction -= with numpy ndarrays
                # or it will also mess up grad_i (!)
        pcgrad_gradients.append(pcgrad_i)

    pcgrad_output = sum(pcgrad_gradients)
    return pcgrad_output


def apply_pcgrad(
        network: Module,
        optimizer: Optimizer,
        task_losses: Mapping[str, tc.Tensor],
        normalize: bool = False) -> None:
    """
    Implements the PCGrad gradient surgery algorithm, and writes the result
    into the network parameters' grad attributes.

    Reference:
        T. Yu et al., 2020 -
            'Gradient Surgery for Multi-Task Learning'

    Args:
        network (torch.nn.Module): Torch Module instance.
        optimizer (torch.optim.Optimizer): Torch Optimizer instance.
        task_losses (Mapping[str, torch.Tensor]): Dictionary mapping from
            task names to losses.
        normalize (bool): Normalize the task gradients before performing PCGrad?
            Default: False.

    Returns:
        None.
    """
    task_gradients = task_losses_to_grads(
        network=network,
        optimizer=optimizer,
        task_losses=task_losses,
        normalize=normalize)
    pcgrad_output = pcgrad_gradient_surgery(task_gradients)
    write_gradient(network, pcgrad_output)
