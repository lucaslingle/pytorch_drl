import abc


class Algo(metaclass=abc.ABCMeta):
    def __init__(self, rank):
        self._rank = rank

    @abc.abstractmethod
    def _annotate(self, trajectory, no_grad, **kwargs):
        """
        Forward pass through the networks.
        """
        pass

    @abc.abstractmethod
    def _credit_assignment(self, trajectory, **kwargs):
        """
        Assign credit backwards in time.
        """
        pass

    @abc.abstractmethod
    def _compute_losses(self, mb, **kwargs):
        """
        Compute losses for learning.
        """
        pass

    @abc.abstractmethod
    def training_loop(self):
        """
        Training loop.
        """
        pass

    @abc.abstractmethod
    def evaluation_loop(self):
        """
        Evaluation loop.
        """
        pass