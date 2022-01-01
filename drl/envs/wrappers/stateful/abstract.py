"""
Abstract wrapper definitions.
"""

import abc

import torch as tc

from drl.envs.wrappers.common import Wrapper
from drl.utils.typing_util import Env, Module, Checkpointable


class StatefulWrapper(Module, Wrapper, metaclass=abc.ABCMeta):
    """
    Wrapper with a checkpointable state.
    """
    def __init__(self, env, **kwargs):
        Wrapper.__init__(self, env)
        Module.__init__(self)

    @abc.abstractmethod
    def get_checkpointables(self):
        """
        in subclasses, permit nesting by doing something like
        checkpointables = {}
        if isinstance(self.env, StatefulWrapper):
             checkpointables.update(self.env.get_checkpointables())
        checkpointables.update({
             f"{self._prefix}_suffix0": self._suffix0,
             f"{self._prefix}_suffix1": self._suffix1,
             ...
        })
        """
        pass


class TrainableWrapper(StatefulWrapper, metaclass=abc.ABCMeta):
    """
    Wrapper with trainable parameters.
    """
    @abc.abstractmethod
    def compute_loss(self, inputs: tc.Tensor, targets: tc.Tensor):
        pass


