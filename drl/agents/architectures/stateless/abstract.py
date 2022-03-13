from typing import Optional, Callable, Mapping, Any, List
import abc

import torch as tc

from drl.agents.architectures.abstract import Architecture


class StatelessArchitecture(Architecture, metaclass=abc.ABCMeta):
    """
    Abstract class for stateless (i.e., memoryless) architectures.
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
            if hasattr(m, 'weight'):
                if self._w_init:
                    self._w_init(m.weight)
            if hasattr(m, 'bias'):
                if self._b_init:
                    self._b_init(m.bias)

    @property
    @abc.abstractmethod
    def input_shape(self) -> List[int]:
        """
        Returns:
            Input shape without batch dimension.
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
        """


class HeadEligibleArchitecture(StatelessArchitecture, metaclass=abc.ABCMeta):
    """
    Abstract class for StatelessArchitecture classes
    that can be used as prediction heads.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            **kwargs: Mapping[str, Any]):
        """
        Args:
            input_dim: Input dimensionality.
                Note that for HeadEligibleArchitectures, the input is assumed
                to be one-dimensional.
            output_dim: Output dimensionality.
            w_init: Weight initializer.
            b_init: Bias initializer.
            **kwargs: Keyword arguments.
        """
        super().__init__(w_init, b_init)
        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def input_shape(self) -> List[int]:
        shape = [self._input_dim]
        return shape

    @property
    def output_dim(self) -> int:
        return self._output_dim
