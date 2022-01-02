import importlib


def get_architecture_cls(cls_name):
    module = importlib.import_module('drl.agents.architectures')
    cls = getattr(module, cls_name)
    return cls


def get_head_cls(cls_name):
    module = importlib.import_module('drl.agents.heads')
    cls = getattr(module, cls_name)
    return cls


def dynamic_mixin(obj, cls, cls_args):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    cls_args_ = {} if not cls_args else cls_args
    obj.__class__ = type(base_cls_name, (base_cls, cls), cls_args_)
