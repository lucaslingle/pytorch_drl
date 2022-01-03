from typing import Dict, List, Any
import importlib

import torch as tc

from drl.envs.wrappers import Wrapper
from drl.agents.preprocessing import Preprocessing, EndToEndPreprocessing
from drl.agents.architectures import Architecture
from drl.agents.heads import Head


def get_class(module_name, cls_name):
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def get_wrapper(env, cls_name, cls_args):
    cls = get_class('drl.envs.wrappers', cls_name)
    return cls(env, **cls_args)


def get_wrappers(env, **wrappers_spec: Dict[str, Dict[str, Any]]):
    for cls_name, cls_args in wrappers_spec.items():
        env = get_wrapper(env, cls_name, cls_args)
    return env


def get_preprocessing(cls_name, cls_args):
    cls = get_class('drl.agents.preprocessing', cls_name)
    return cls(**cls_args)


def get_preprocessings(**preprocessing_spec: Dict[str, Dict[str, Any]]):
    preprocessing_stack = list()
    for cls_name, cls_args in preprocessing_spec.items():
        preprocessing = get_preprocessing(
            cls_name=cls_name, cls_args=cls_args)
        preprocessing_stack.append(preprocessing)
    return EndToEndPreprocessing(tc.nn.Sequential(*preprocessing_stack))


def get_architecture(cls_name, cls_args):
    cls = get_class('drl.agents.architectures', cls_name)
    return cls(**cls_args)


def get_predictor(cls_name, cls_args):
    cls = get_class('drl.agents.heads', cls_name)
    return cls(**cls_args)


def get_predictors(**predictors_spec: Dict[str, Dict[str, Any]]):
    predictors = dict()
    for key, spec in predictors_spec.items():
        predictors[key] = get_predictor(**spec)
    return predictors


def get_optimizer(model, cls_name, cls_args):
    cls = get_class('torch.optim', cls_name)
    return cls(model.parameters(), **cls_args)


def get_scheduler(optimizer, cls_name, cls_args):
    if cls_name == 'None':
        return None
    cls = get_class('torch.optim.lr_scheduler', cls_name)
    return cls(optimizer, **cls_args)


class Agent(tc.nn.Module):
    def __init__(
            self,
            preprocessing: Preprocessing,
            architecture: Architecture,
            predictors: Dict[str, Head]
    ):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._predictors = predictors

    def forward(self, x: tc.Tensor, predict: List[str]) -> Dict[str, tc.Tensor]:
        preproc = self._preprocessing(x)
        features = self._architecture(preproc)
        predictions = {
            key: self._predictors[key](features) for key in predict
        }
        return predictions
