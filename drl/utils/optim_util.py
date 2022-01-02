"""
Learning rate scheduler util.
"""

from typing import Optional, Dict, Any
import importlib

from drl.utils.typing_util import Module, Optimizer, Scheduler


def get_optimizer(
        model: Module,
        optimizer_cls_name: str,
        optimizer_args: Dict[str, Any]
) -> Optimizer:
    module = importlib.import_module('torch.optim')
    optimizer_cls = getattr(module, optimizer_cls_name)
    return optimizer_cls(model.parameters(), **optimizer_args)


def get_scheduler(
        optimizer: Optimizer,
        scheduler_cls_name: str,
        scheduler_args: Dict[str, Any]
) -> Optional[Scheduler]:
    if scheduler_cls_name == 'None':
        return None
    module = importlib.import_module('torch.optim.lr_scheduler')
    scheduler_cls = getattr(module, scheduler_cls_name)
    return scheduler_cls(optimizer, **scheduler_args)