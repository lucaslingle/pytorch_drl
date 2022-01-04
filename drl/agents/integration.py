from typing import Dict, List, Any
import importlib

import torch as tc

from drl.agents.preprocessing import Preprocessing, EndToEndPreprocessing
from drl.agents.architectures import Architecture
from drl.agents.heads import (
    Head, CategoricalActionValueHead, EpsilonGreedyCategoricalPolicyHead
)


def get_preprocessing(cls_name, cls_args):
    module = importlib.import_module('drl.agents.preprocessing')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_preprocessings(**preprocessing_spec: Dict[str, Dict[str, Any]]):
    preprocessing_stack = list()
    for cls_name, cls_args in preprocessing_spec.items():
        preprocessing = get_preprocessing(
            cls_name=cls_name, cls_args=cls_args)
        preprocessing_stack.append(preprocessing)
    return EndToEndPreprocessing(tc.nn.Sequential(*preprocessing_stack))


def get_architecture(cls_name, cls_args):
    module = importlib.import_module('drl.agents.architectures')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_predictor(cls_name, cls_args):
    module = importlib.import_module('drl.agents.heads')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_predictors(**predictors_spec: Dict[str, Dict[str, Any]]):
    predictors = dict()
    for key, spec in predictors_spec.items():
        predictor = get_predictor(**spec)
        if isinstance(predictor, CategoricalActionValueHead):
            policy_predictor = EpsilonGreedyCategoricalPolicyHead(
                action_value_head=predictor)
            predictors['policy'] = policy_predictor
        predictors[key] = predictor
    return predictors


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

    @property
    def keys(self):
        return self._predictors.keys()

    def forward(
            self,
            x: tc.Tensor,
            predict: List[str],
            **kwargs: Dict[str, Any]
    ) -> Dict[str, tc.Tensor]:
        """
        Args:
            x: Batch of observations
            predict: Names of predictors to apply.
            **kwargs: Keyword arguments to pass to predictors.
        Returns:
            Dictionary of predictions.
        """
        preproc = self._preprocessing(x)
        features = self._architecture(preproc)
        predictions = {
            key: self._predictors[key](features, **kwargs) for key in predict
        }
        return predictions
