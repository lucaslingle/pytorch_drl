from typing import Optional, Callable, Mapping, Any, List
import abc

import torch as tc

from drl.agents.architectures.abstract import Architecture


class StatefulArchitecture(Architecture, metaclass=abc.ABCMeta):
    """
    Abstract class for stateful (i.e., memory-augmented) architectures.
    """
    def __init__(
            self,
            w_init: Optional[Callable[[tc.Tensor], None]],
            b_init: Optional[Callable[[tc.Tensor], None]],
            **kwargs: Mapping[str, Any]):
        """
        Args:
            w_init (Optional[Callable[[torch.Tensor], None]]): Weight initializer.
            b_init (Optional[Callable[[torch.Tensor], None]]): Bias initializer.
            **kwargs (Mapping[str, Any]): Keyword arguments.
        """
        super().__init__()
        self._w_init = w_init
        self._b_init = b_init

    def _init_weights(self, sequential_module: tc.nn.Sequential) -> None:
        for m in sequential_module:
            if hasattr(m, 'weights'):
                if self._w_init:
                    self._w_init(m.weights)
            if hasattr(m, 'bias'):
                if self._b_init:
                    self._b_init(m.bias)

    @property
    @abc.abstractmethod
    def input_shape(self) -> List[int]:
        """
        Returns:
            Input shape without batch or time dimension.
        """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """
        Returns:
            Dimensionality of output features.
        """

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        """
        Forward method.

        If x is two-dimensional, the first axis is assumed to be the batch axis,
        and the time axis is then expanded automatically.
        """
