"""
Abstract wrapper definitions.
"""

import abc

from drl.envs.wrappers.stateless import Wrapper


class TrainableWrapper(Wrapper, metaclass=abc.ABCMeta):
    """
    Wrapper with trainable parameters.
    """
    @abc.abstractmethod
    def learn(self, obs_batch, **kwargs):
        pass
