from typing import Mapping, Any, Type, Callable, Dict
import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture


class ValueHead(Head, metaclass=abc.ABCMeta):
    """
    Value head abstract class.
    """


# default for ppo was architecture_cls_name=Linear, w_init=('normc', {'gain': 1.0}), b_init=('zeros_', {})
class SimpleValueHead(ValueHead):
    """
    Simple value prediction head.
    """
    def __init__(
            self,
            num_features: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Dict[str, Any],
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            num_features: Number of input features.
            head_architecture_cls: Class name for policy head architecture.
                Must be a derived class of HeadEligibleArchitecture.
            head_architecture_cls_args: Keyword arguments for head architecture.
            w_init: Weight initializer.
            b_init: Bias initializer.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._value_head = head_architecture_cls.__init__(
            input_dim=num_features,
            output_dim=1,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def forward(self, features, **kwargs):
        vpred = self._value_head(features).squeeze(-1)
        return vpred
