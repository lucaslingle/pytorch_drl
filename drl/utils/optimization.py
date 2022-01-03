import importlib


def get_optimizer(model, cls_name, cls_args):
    module = importlib.import_module('torch.optim')
    cls = getattr(module, cls_name)
    return cls(model.parameters(), **cls_args)


def get_scheduler(optimizer, cls_name, cls_args):
    if cls_name == 'None':
        return None
    module = importlib.import_module('torch.optim.lr_scheduler')
    cls = getattr(module, cls_name)
    return cls(optimizer, **cls_args)