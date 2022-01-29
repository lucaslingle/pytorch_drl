"""
Abstract wrapper definitions.
"""

from typing import Mapping, Any
import abc

import torch as tc

from drl.envs.wrappers.stateless import Wrapper


class TrainableWrapper(Wrapper, metaclass=abc.ABCMeta):
    """
    Wrapper with trainable parameters.
    """
    @abc.abstractmethod
    def learn(self, mb: Mapping[str, tc.Tensor], **kwargs: Mapping[str, Any]):
        """
        Args:
            mb (Mapping[str, torch.Tensor]): Minibatch of experience to learn from.
            **kwargs (Mapping[str, Any]):

        Returns:
            None.
        """
        # todo(lucaslingle):
        #   make this return a possibly-empty dictionary of tensorboard metrics.
        raise NotImplementedError
