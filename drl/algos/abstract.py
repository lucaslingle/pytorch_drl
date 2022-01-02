import abc


class Algo(metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config

    @abc.abstractmethod
    def _get_learning_system(self):
        pass

    @abc.abstractmethod
    def _compute_losses(self, mb):
        pass

    @abc.abstractmethod
    def _train_loop(self):
        pass

    @abc.abstractmethod
    def _evaluation_loop(self):
        pass

    @abc.abstractmethod
    def _maybe_load_checkpoints(self):
        pass

    @abc.abstractmethod
    def _save_checkpoints(self):
        pass
