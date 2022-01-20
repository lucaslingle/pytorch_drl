import random

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import gym

from drl.envs.wrappers.integration import get_wrappers
from drl.agents.integration import (
    Agent, get_preprocessings, get_architecture, get_predictors
)
from drl.utils.optimization import get_optimizer, get_scheduler
from drl.utils.checkpointing import maybe_load_checkpoints


def _get_process_seed(rank, config):
    return config['seed'] + 10000 * rank


def _set_seed(process_seed):
    tc.manual_seed(process_seed)
    np.random.seed(process_seed % 2 ** 32)
    random.seed(process_seed % 2 ** 32)


def _get_env(env_config, process_seed, mode):
    env = gym.make(env_config.get('id'))
    env.seed(process_seed)
    wrapper_config = env_config.get('wrappers')
    if mode == 'train':
        mode_wrappers = wrapper_config.get('train')
    else:
        mode_wrappers = wrapper_config.get('evaluate')
    env = get_wrappers(env, **mode_wrappers)
    return env


def _get_net(net_config, env, rank):
    preprocessing = get_preprocessings(**net_config.get('preprocessing'))
    architecture = get_architecture(**net_config.get('architecture'))
    predictors = get_predictors(rank, env, **net_config.get('predictors'))
    return DDP(Agent(preprocessing, architecture, predictors))


def _get_opt(opt_config, agent):
    optimizer = get_optimizer(model=agent, **opt_config)
    return optimizer


def _get_sched(sched_config, optimizer):
    scheduler = get_scheduler(optimizer=optimizer, **sched_config)
    return scheduler


def make_learning_system(rank, config):
    """
    Provides a simple, flexible, and reproducible launch framework
        for the drl library.

    For use in conjunction with the configuration files provided in models_dir.

    Args:
        rank: Process rank.
        config: drl.utils.ConfigParser instance.

    Returns:
        Dictionary with environment, networks, optimizers, schedulers,
        and global step of the learning process thus far.
    """
    mode = config.get('mode')

    process_seed = _get_process_seed(rank, config)
    _set_seed(process_seed)

    env_config = config.get('env')
    env = _get_env(env_config, process_seed, mode)

    learners_config = config.get('networks')
    learning_system = dict()
    for prefix in learners_config:
        learner_config = learners_config.get(prefix)
        for suffix in learner_config:
            name = f'{prefix}_{suffix}'
            if suffix == 'net':
                net_config = learner_config.get(suffix)
                net = _get_net(net_config, env, rank)
                learning_system[name] = net
            elif suffix == 'optimizer':
                opt_config = learner_config.get(suffix)
                opt = _get_opt(opt_config, learning_system[f'{prefix}_net'])
                learning_system[name] = opt
            elif suffix == 'scheduler':
                sched_config = learner_config.get(suffix)
                sched = _get_sched(
                    sched_config, learning_system[f'{prefix}_optimizer'])
                learning_system[name] = sched
            elif suffix == 'use_shared_architecture':
                learning_system[f'{prefix}_net'] = None
                learning_system[f'{prefix}_optimizer'] = None
                learning_system[f'{prefix}_scheduler'] = None
                break
            else:
                msg = f'Unrecognized suffix {suffix} in config file.'
                raise ValueError(msg)

        checkpointables = {k: v for k,v in learning_system.items()}
        checkpointables.update(env.get_checkpointables())
        global_step = maybe_load_checkpoints(
            checkpoint_dir=config.get('checkpoint_dir'),
            checkpointables=checkpointables,
            map_location='cpu',
            steps=None)
        return {'global_step': global_step, 'env': env, **learning_system}
