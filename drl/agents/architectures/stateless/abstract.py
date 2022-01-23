import abc

from drl.agents.architectures.abstract import Architecture


class StatelessArchitecture(Architecture, metaclass=abc.ABCMeta):
    """
    Abstract class for stateless (i.e., memoryless) architectures.
    """
    def __init__(self, w_init, b_init, **kwargs):
        super().__init__()
        self._w_init = w_init
        self._b_init = b_init

    def _init_weights(self, sequential_module):
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
            Input shape without batch dimension.
        """

    @property
    @abc.abstractmethod
    def output_dim(self):
        """
        Returns:
            Dimensionality of output features.
        """


class HeadEligibleArchitecture(StatelessArchitecture):
    def __init__(self, input_dim, output_dim, w_init, b_init, **kwargs):
        super().__init__(w_init, b_init)
        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def input_shape(self):
        tuple = (self._input_dim,)
        return tuple

    @property
    def output_dim(self):
        return self._output_dim
