import abc

import torch as tc

from drl.agents.preprocessing.abstract import Preprocessing


class Architecture(tc.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, preprocessing: Preprocessing):
        super().__init__()
        self._preprocessing = preprocessing
