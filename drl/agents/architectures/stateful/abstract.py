from typing import Callable, Mapping, Any
import abc

import torch as tc

from drl.agents.architectures.abstract import Architecture


class StatefulArchitecture(Architecture, metaclass=abc.ABCMeta):
    """
    Abstract class for stateful (i.e., memory-augmented) architectures.
    """
    def __init__(
            self,
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            w_init: Weight initializer.
            b_init: Bias initializer.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._w_init = w_init
        self._b_init = b_init

    def _init_weights(self, sequential_module: tc.nn.Sequential) -> None:
        for m in sequential_module:
            if hasattr(m, 'weights'):
                self._w_init(m.weights)
            if hasattr(m, 'bias'):
                self._b_init(m.bias)

    @property
    @abc.abstractmethod
    def input_shape(self):
        """
        Returns:
            Input shape without batch or time dimension.
        """

    @property
    @abc.abstractmethod
    def output_dim(self):
        """
        Returns:
            Dimensionality of output features.
        """
