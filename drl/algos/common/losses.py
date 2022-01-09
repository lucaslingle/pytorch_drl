import importlib


def get_loss(cls_name):
    module = importlib.import_module('torch.nn')
    cls = getattr(module, cls_name)
    obj = cls(reduction='none')
    return obj
