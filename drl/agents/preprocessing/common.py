from typing import Dict, Any
import importlib

import torch as tc

from drl.agents.preprocessing.abstract import Preprocessing


def get_preprocessing(
        module_name: str,
        cls_name: str,
        cls_args: Dict[str, Any]
) -> Preprocessing:
    module = importlib.import_module(module_name)
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
        for fq_cls_name, cls_args in self._preprocessing_spec:
            module_name, _, cls_name = fq_cls_name.rpartition(".")
            preprocessing = get_preprocessing(
                module_name=module_name,
                cls_name=cls_name,
                cls_args=cls_args)
            preprocessing_stack.append(preprocessing)
        return tc.nn.Sequential(*preprocessing_stack)

    def forward(self, x):
        return self._preprocessing(x)
