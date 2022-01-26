from typing import Mapping

import numpy as np
import torch as tc
from  torch.nn.parallel import DistributedDataParallel as DDP

from drl.agents.integration import Agent
from drl.utils.types import Optimizer


def norm(vector: np.ndarray) -> float:
    return np.sqrt(np.sum(np.square(vector)) + 1e-6)


@tc.no_grad()
def read_gradient(network: DDP, normalize: bool) -> np.ndarray:
    """
    Reads the currently stored gradient in the network parameters' grad
    attributes, and returns it as a vector.

    Args:
        network (DDP): DDP-wrapped `Agent` instance.
        normalize (bool): Whether to normalize the gradient extracted.

    Returns:
        numpy.ndarray: Gradient as a vector.
    """
    gradient_subvecs = []
    for p in network.parameters():
        if p.grad is not None:
            subvec = p.grad.reshape(-1).detach().numpy()
            gradient_subvecs.append(subvec)
    gradient = np.concatenate(gradient_subvecs, axis=0)
    if normalize:
        gradient /= norm(gradient)
    return gradient


@tc.no_grad()
def write_gradient(network: DDP, gradient: np.ndarray) -> None:
    """
    Writes a gradient vector into a network's parameters' grad attributes.

    Args:
        network (DDP): Agent instance.
        gradient (numpy,ndarray): Gradient as a numpy ndarray.

    Returns:
        None.
    """
    dims_so_far = 0
    for p in network.parameters():
        if p.grad is not None:
            numel = np.prod(p.shape)
            subvec = gradient[dims_so_far:dims_so_far+numel]
            p.grad.copy_(tc.tensor(subvec, requires_grad=False).reshape(p.shape))
            dims_so_far += numel


def pcgrad_gradient_surgery(
        network: DDP,
        optimizer: Optimizer,
        task_losses: Mapping[str, tc.Tensor],
        normalize: bool = True
) -> np.ndarray:
    """
    Implements the PCGrad gradient surgery algorithm.

    Reference:
        T. Yu et al., 2020 -
            'Gradient Surgery for Multi-Task Learning'

    Args:
        network (DDP): DDP-wrapped `Agent` instance.
        optimizer (torch.optim.Optimizer): Torch Optimizer instance.
        task_losses (Mapping[str, torch.Tensor]): Dictionary of losses to
            perform PCGrad on, keyed by name.
        normalize (bool): Normalize the task gradients before performing PCGrad?
            Default: True.

    Returns:
        numpy.ndarray: PCGrad gradient as a vector.
    """
    task_gradients = dict()
    for k in task_losses:
        optimizer.zero_grad()
        task_losses[k].backward(retain_graph=True)
        task_gradients[k] = read_gradient(network, normalize)

    pcgrad_gradients = list()
    for i in task_losses:
        grad_i = task_gradients[i]
        pcgrad_i = grad_i
        for j in task_losses:
            if i == j:
                continue
            grad_j = task_gradients[j]
            if np.dot(pcgrad_i, grad_j) < 0.:
                coef = np.dot(pcgrad_i, grad_j) / np.square(norm(grad_j))
                pcgrad_i -= coef * grad_j
        pcgrad_gradients.append(pcgrad_i)

    optimizer.zero_grad()
    pcgrad_output = sum(pcgrad_gradients)
    return pcgrad_output


def apply_pcgrad(
        network: DDP,
        optimizer: Optimizer,
        task_losses: Mapping[str, tc.Tensor],
        normalize: bool = True
) -> None:
    """
    Implements the PCGrad gradient surgery algorithm, and writes the result
    into the network parameters' grad attributes.

    Reference:
        T. Yu et al., 2020 -
            'Gradient Surgery for Multi-Task Learning'

    Args:
        network (DDP): DDP-wrapped `Agent` instance.
        optimizer (torch.optim.Optimizer): Torch Optimizer instance.
        task_losses (Mapping[str, torch.Tensor]): Dictionary of losses to
            perform PCGrad on, keyed by name.
        normalize (bool): Normalize the task gradients before performing PCGrad?
            Default: True.

    Returns:
        None.
    """
    pcgrad_output = pcgrad_gradient_surgery(
        network, optimizer, task_losses, normalize)
    write_gradient(network, pcgrad_output)
