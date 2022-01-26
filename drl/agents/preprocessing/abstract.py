import abc

import torch as tc


class Preprocessing(tc.nn.Module, metaclass=abc.ABCMeta):
    """
    Runs preprocessing ops.
    """

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        """
        Forward method.
        """
