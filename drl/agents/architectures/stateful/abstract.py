import abc

from drl.agents.architectures.abstract import Architecture
from drl.agents.architectures.common import get_initializer


class StatefulArchitecture(Architecture, metaclass=abc.ABCMeta):
    """
    Abstract class for stateful (i.e., memory-augmented) architectures.
    """
    def __init__(self, w_init_spec, b_init_spec):
        super().__init__()
        self._w_init = get_initializer(name=w_init_spec[0])(**w_init_spec[1])
        self._b_init = get_initializer(name=b_init_spec[0])(**b_init_spec[1])

    def _init_weights(self):
        for m in self._network:
            self._w_init(m.weights)
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
