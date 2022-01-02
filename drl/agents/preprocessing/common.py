from typing import Dict, Any
import importlib

import torch as tc

from drl.agents.preprocessing.abstract import Preprocessing


def get_preprocessing(
        cls_name: str,
        cls_args: Dict[str, Any]
) -> Preprocessing:
    module = importlib.import_module('drl.agents.preprocessing')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


class EndToEndPreprocessing(Preprocessing):
    def __init__(self, preprocessing_spec: Dict[str, Dict[str, Any]]):
        """
        Args:
            preprocessing_spec: Dictionary of preprocessor class arguments,
                keyed by fully-qualified preprocessor class name.
        """
        super().__init__()
        self._preprocessing_spec = preprocessing_spec
        self._preprocessing = self._build()

    def _build(self):
        preprocessing_stack = list()
        for cls_name, cls_args in self._preprocessing_spec.items():
            preprocessing = get_preprocessing(
                cls_name=cls_name, cls_args=cls_args)
            preprocessing_stack.append(preprocessing)
        return tc.nn.Sequential(*preprocessing_stack)

    def forward(self, x):
        return self._preprocessing(x)
