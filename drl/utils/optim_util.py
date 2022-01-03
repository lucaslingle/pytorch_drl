"""
Optimization util.
"""

from typing import Optional, Dict, Any
import importlib

from drl.utils.typing_util import Module, Optimizer, Scheduler


def get_optimizer(
        model: Module,
        cls_name: str,
        cls_args: Dict[str, Any]
) -> Optimizer:
    module = importlib.import_module('torch.optim')
    cls = getattr(module, cls_name)
    return cls(model.parameters(), **cls_args)


def get_scheduler(
        optimizer: Optimizer,
        cls_name: str,
        cls_args: Dict[str, Any]
) -> Optional[Scheduler]:
    if cls_name == 'None':
        return None
    module = importlib.import_module('torch.optim.lr_scheduler')
    cls = getattr(module, cls_name)
    return cls(optimizer, **cls_args)
