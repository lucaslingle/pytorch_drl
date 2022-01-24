import importlib

import torch as tc


def get_loss(cls_name: str) -> tc.nn.modules.loss._Loss:
    """
    Args:
        cls_name: Class name for a derived class of torch.nn.modules.loss._Loss.

    Returns:
        Instantiated loss class.
    """
    module = importlib.import_module('torch.nn')
    cls = getattr(module, cls_name)
    obj = cls(reduction='none')
    return obj
