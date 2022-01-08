from typing import Dict

import numpy as np
import torch as tc

from drl.agents.integration import Agent
from drl.utils.types import Optimizer


def norm(vector: np.ndarray) -> float:
    return np.sqrt(np.sum(np.square(vector)) + 1e-6)


@tc.no_grad()
def extract_gradient(network: Agent, normalize: bool) -> np.ndarray:
    gradient_subvecs = []
    for p in network.params():
        if p.grad is not None:
            subvec = p.grad.reshape(-1).detach().numpy()
        else:
            numel = np.prod(p.shape)
            subvec = np.zeros((numel,), dtype=tc.float32)
        gradient_subvecs.append(subvec)
    gradient = np.concatenate(gradient_subvecs, dim=0)
    if normalize:
        gradient /= norm(gradient)
    return gradient


def pcgrad_gradient_surgery(
        network: Agent,
        optimizer: Optimizer,
        task_losses: Dict[str, tc.Tensor],
        normalize: bool = True
) -> np.ndarray:
    pcgrad_gradients = []
    for i in task_losses:
        optimizer.zero_grad()
        task_losses[i].backward()
        grad_i = extract_gradient(network, normalize)
        pcgrad_i = grad_i
        for j in task_losses:
            if i == j:
                continue
            optimizer.zero_grad()
            task_losses[i].backward()
            grad_j = extract_gradient(network, normalize)
            if np.dot(pcgrad_i, grad_j) < 0.:
                coef = np.dot(pcgrad_i, grad_j) / np.square(norm(grad_j))
                pcgrad_i -= coef * grad_j
        pcgrad_gradients.append(pcgrad_i)
    optimizer.zero_grad()
    pcgrad_output = sum(pcgrad_gradients)
    return pcgrad_output


@tc.no_grad()
def write_gradient(network: Agent, gradient: np.ndarray) -> None:
    dims_so_far = 0
    for p in network.params():
        numel = np.prod(p.shape)
        subvec = gradient[dims_so_far:dims_so_far+numel]
        p.grad.copy_(tc.tensor(subvec, requires_grad=False).reshape(p.shape))
        dims_so_far += numel


def apply_pcgrad(
        network: Agent,
        optimizer: Optimizer,
        task_losses: Dict[str, tc.Tensor],
        normalize: bool = True
) -> None:
    pcgrad_output = pcgrad_gradient_surgery(
        network, optimizer, task_losses, normalize)
    write_gradient(network, pcgrad_output)
