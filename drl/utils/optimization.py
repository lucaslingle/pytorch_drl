from typing import Dict, Any, Optional, List, Union
import importlib

import torch as tc

from drl.utils.types import Module, Optimizer, Scheduler


def get_optimizer(
        model: Module,
        cls_name: str,
        cls_args: Dict[str, Any]
) -> Optimizer:
    """
    Creates an optimizer.

    Args:
        model (Module): Torch module instance.
        cls_name (str): Optimizer class name. Must correspond to a derived class
            of torch.optim.Optimizer.
        cls_args (Dict[str, Any]): Dictionary of arguments passed to the
            class constructor.

    Returns:
        torch.optim.Optimizer: Instantiated optimizer.

    Note:
        If the cls_name is 'AdamW', our implementation only applies weight decay
        to the weights, and not biases/normalization parameters. To disable this,
        you can use cls_name 'Adam' instead.
    """
    module = importlib.import_module('torch.optim')
    cls = getattr(module, cls_name)
    if cls_name == 'AdamW':
        wd = cls_args.pop('wd')
        param_groups = get_weight_decay_param_groups(model, wd)
        return cls(param_groups, **cls_args)
    return cls(model.parameters(), **cls_args)


def get_scheduler(
        optimizer: Optimizer,
        cls_name: str,
        cls_args: Dict[str, Any]
) -> Optional[Scheduler]:
    """
    Creates a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Torch optimizer to wrap.
        cls_name (str): Scheduler class name, which should correspond
            to a derived class of torch.optim.lr_scheduler._LRScheduler,
            or 'None'.
        cls_args: Dictionary of arguments passed to class constructor.

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Instantiated scheduler
            or None.
    """
    if cls_name == 'None':
        return None
    module = importlib.import_module('torch.optim.lr_scheduler')
    cls = getattr(module, cls_name)
    return cls(optimizer, **cls_args)


def get_weight_decay_param_groups(
        model: Module,
        wd: float
) -> List[Dict[str, Union[tc.nn.parameter.Parameter, float]]]:
    """
    Splits a model's parameters into two parameter groups:
        those to apply weight decay to, and those not to.

    Args:
        model (torch.nn.Module): Torch module instance.
        wd (float): Weight decay coefficient.

    Returns:
        List[Dict[str, Union[tc.nn.parameter.Parameter, float]],
           Dict[str, Union[tc.nn.parameter.Parameter, float]]]:
           List of two dictionaries with keys 'params', and 'weight_decay'.
    """
    apply_decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1:
            no_decay.append(param)
        else:
            apply_decay.append(param)
    return [
        {'params': apply_decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
