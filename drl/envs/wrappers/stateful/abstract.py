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
    def learn(
            self, minibatch: Mapping[str, tc.Tensor],
            **kwargs: Mapping[str, Any]) -> None:
        """
        Args:
            minibatch (Mapping[str, torch.Tensor]): Minibatch of experience.
            **kwargs (Mapping[str, Any]): Keyword args.

        Returns:
            None.
        """
        # todo(lucaslingle):
        #   make this return a possibly-empty dictionary of tensorboard metrics.
        raise NotImplementedError
