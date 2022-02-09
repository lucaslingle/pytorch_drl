"""
Value prediction heads.
"""

from typing import Mapping, Any, Type, Callable, Optional
import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture


class ValueHead(Head, metaclass=abc.ABCMeta):
    """
    Value head abstract class.
    """


class SimpleValueHead(ValueHead):
    """
    Simple value prediction head.
    """
    def __init__(
            self,
            num_features: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Mapping[str, Any],
            w_init: Optional[Callable[[tc.Tensor], None]],
            b_init: Optional[Callable[[tc.Tensor], None]],
            **kwargs: Mapping[str, Any]):
        """
        Args:
            num_features (int): Number of input features.
            head_architecture_cls (Type[HeadEligibleArchitecture]): Class object
                for policy head architecture. Must be a derived class of
                HeadEligibleArchitecture.
            head_architecture_cls_args (Mapping[str, Any]): Keyword arguments
                for head architecture.
            w_init (Callable[[torch.Tensor], None]): Weight initializer.
            b_init (Callable[[torch.Tensor], None]): Bias initializer.
            **kwargs (Mapping[str, Any]): Keyword arguments.
        """
        super().__init__()
        self._value_head = head_architecture_cls(
            input_dim=num_features,
            output_dim=1,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def forward(
            self, features: tc.Tensor, **kwargs: Mapping[str,
                                                         Any]) -> tc.Tensor:
        """
        Args:
            features (torch.Tensor): Torch tensor with shape [batch_size, num_features].
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            torch.Tensor: Torch tensor of shape [batch_size], containing the
                estimated state-conditional values.
        """
        vpred = self._value_head(features).squeeze(-1)
        return vpred
