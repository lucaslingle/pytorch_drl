"""
Abstract wrapper definitions.
"""

from typing import Dict, Any
import abc

import torch as tc

from drl.envs.wrappers.common import Wrapper
from drl.utils.typing_util import Env, Module


class StatefulWrapper(Module, Wrapper, metaclass=abc.ABCMeta):
    """
    Wrapper with a checkpointable state.
    """
    def __init__(self, env: Env, **kwargs: Dict[str, Any]):
        Wrapper.__init__(self, env)
        Module.__init__(self, **kwargs)


class TrainableWrapper(StatefulWrapper, metaclass=abc.ABCMeta):
    """
    Wrapper with trainable parameters.
    """
    def __init__(self, env: Env, **kwargs: Dict[str, Any]):
        Wrapper.__init__(self, env)
        Module.__init__(self, **kwargs)

    @abc.abstractmethod
    def compute_loss(self, inputs: tc.Tensor, targets: tc.Tensor):
        pass
