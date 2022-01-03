import abc

from drl.utils.checkpointing import save_checkpoints, maybe_load_checkpoints


class Algo(metaclass=abc.ABCMeta):
    def __init__(self, rank, config):
        self._rank = rank
        self._config = config

    @abc.abstractmethod
    def _get_learning_system(self):
        pass

    @abc.abstractmethod
    def _compute_losses(self, mb):
        pass

    @abc.abstractmethod
    def training_loop(self):
        pass

    @abc.abstractmethod
    def evaluation_loop(self):
        pass

    def _save_checkpoints(self, checkpointables, step):
        save_checkpoints(
            checkpoint_dir=self._config.get('checkpoint_dir'),
            checkpointables=checkpointables,
            steps=step)

    def _maybe_load_checkpoints(self, checkpointables, step):
        global_step = maybe_load_checkpoints(
            checkpoint_dir=self._config.get('checkpoint_dir'),
            checkpointables=checkpointables,
            map_location='cpu',
            steps=step)
        return global_step