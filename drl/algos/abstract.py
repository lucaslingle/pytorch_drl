import abc

import gym
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.envs.wrappers.integration import get_wrappers
from drl.agents.integration import (
    Agent, get_preprocessings, get_architecture, get_predictors
)
from drl.utils.optimization import get_optimizer
from drl.utils.checkpointing import save_checkpoints, maybe_load_checkpoints


class Algo(metaclass=abc.ABCMeta):
    def __init__(self, rank, config):
        self._rank = rank
        self._config = config

    @staticmethod
    def _get_net(net_config):
        preprocessing = get_preprocessings(**net_config.get('preprocessing'))
        architecture = get_architecture(**net_config.get('architecture'))
        predictors = get_predictors(**net_config.get('predictors'))
        return DDP(Agent(preprocessing, architecture, predictors))

    @staticmethod
    def _get_opt(opt_config, agent):
        optimizer = get_optimizer(model=agent, **opt_config)
        return optimizer

    @staticmethod
    def _get_env(env_config):
        env = gym.make(env_config.get('id'))
        env = get_wrappers(env=env, **env_config.get('wrappers'))
        return env

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

    @abc.abstractmethod
    def _get_learning_system(self):
        pass

    @abc.abstractmethod
    def _annotate(self):
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