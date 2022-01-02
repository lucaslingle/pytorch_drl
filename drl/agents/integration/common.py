import importlib

import torch as tc


def get_architecture_cls(cls_name):
    module = importlib.import_module('drl.agents.architectures')
    cls = getattr(module, cls_name)
    return cls


def get_architecture(cls_name, cls_args, preprocessing):
    cls = get_architecture_cls(cls_name)
    return cls(preprocessing=preprocessing, **cls_args)


def get_head_cls(cls_name):
    module = importlib.import_module('drl.agents.heads')
    cls = getattr(module, cls_name)
    return cls


def dynamic_mixin(obj, cls, cls_args):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls), **cls_args)


class IntegratedAgent(tc.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        preprocessed = self.preprocessed(x)
        features = self.features(preprocessed)
        return features
