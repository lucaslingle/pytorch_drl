import abc

from drl.agents.architectures.abstract import Architecture
from drl.agents.architectures.common import get_initializer


class StatelessArchitecture(Architecture, metaclass=abc.ABCMeta):
    """
    Abstract class for stateless (i.e., memoryless) architectures.
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
            Input shape without batch dimension.
        """

    @property
    @abc.abstractmethod
    def output_dim(self):
        """
        Returns:
            Dimensionality of output features.
        """
