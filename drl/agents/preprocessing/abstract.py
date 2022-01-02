import abc

import torch as tc


class Preprocessing(tc.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x):
        pass
