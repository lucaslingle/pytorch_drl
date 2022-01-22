from typing import Tuple, Mapping, Any
import abc

from drl.agents.heads.abstract import Head
from drl.agents.integration import get_architecture


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
            architecture_cls_name: str,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]],
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            num_features: Number of input features.
            architecture_cls_name: Class name for policy head architecture.
                Must be a derived class of StatelessArchitecture.
            w_init_spec: Tuple containing weight initializer name and kwargs.
            b_init_spec: Tuple containing bias initializer name and kwargs.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._value_head = get_architecture(
            cls_name=architecture_cls_name,
            cls_args={
                'input_dim': num_features,
                'output_dim': 1,
                'w_init_spec': w_init_spec,
                'b_init_spec': b_init_spec
            })

    def forward(self, features, **kwargs):
        vpred = self._value_head(features).squeeze(-1)
        return vpred
