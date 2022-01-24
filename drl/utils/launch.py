"""
Optional launch utility -
    for use with provided script and specified config file format.
"""

from typing import List, Any, Mapping, Union, Type, Tuple, Optional
import random
import importlib

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import gym

from drl.utils.configuration import ConfigParser
from drl.envs.wrappers import Wrapper
from drl.agents.preprocessing.abstract import Preprocessing
from drl.agents.architectures import Architecture
from drl.agents.architectures.stateless.abstract import StatelessArchitecture
from drl.agents.architectures.stateful.abstract import StatefulArchitecture
from drl.utils.initializers import get_initializer
from drl.agents.heads import (
    Head,
    DiscreteActionValueHead,
    EpsilonGreedyCategoricalPolicyHead,
    DiscretePolicyHead,
    ContinuousPolicyHead
)
from drl.agents.integration.agent import Agent
from drl.utils.optimization import get_optimizer, get_scheduler
from drl.utils.checkpointing import maybe_load_checkpoints
from drl.utils.types import Optimizer, Scheduler


def get_process_seed(rank: int, config: ConfigParser) -> int:
    """
    Computes a process-specific RNG seed to ensure experiential diversity
    and per-machine experimental reproducibility.

    Args:
        rank (int): Process rank.
        config (`ConfigParser`): Configuration object.

    Returns:
        int: Process seed.
    """
    return config['seed'] + 10000 * rank


def set_seed(process_seed: int) -> None:
    """
    Sets the random number generators on this process to the provided RNG seed.

    Args:
        process_seed (int): RNG seed integer.

    Returns:
        None.
    """
    tc.manual_seed(process_seed)
    np.random.seed(process_seed % 2 ** 32)
    random.seed(process_seed % 2 ** 32)


def get_wrapper(
        env: Union[gym.core.Env, Wrapper],
        cls_name: str,
        cls_args: Mapping[str, Any]
) -> Wrapper:
    """
    Args:
        env (Union[gym.core.Env, `Wrapper`]): OpenAI gym environment or
            existing `Wrapper' to be further wrapped.
        cls_name (str): Wrapper class name.
        cls_args (Mapping[str, Any]): Dictionary of wrapper constructor args.

    Returns:
        Wrapper: Wrapped environment.
    """
    module = importlib.import_module('drl.envs.wrappers')
    cls = getattr(module, cls_name)
    return cls(env, **cls_args)


def get_wrappers(
        env: Union[gym.core.Env, Wrapper],
        **wrappers_spec: Mapping[str, Mapping[str, Any]]
) -> Wrapper:
    """
    Args:
        env (Union[gym.core.Env, `Wrapper`]): OpenAI gym environment or `Wrapper` thereof.
        **wrappers_spec (Mapping[str, Mapping[str, Any]]): Dictionary of all wrappers to apply.
            Python dictionaries are not inherently ordered, but here the
            ordering is assumed to be correct, since in our script pyyaml
            builds it by parsing the file sequentially.

    Returns:
        Union[gym.core.Env, Wrapper]: Wrapped environment.
    """
    for cls_name, cls_args in wrappers_spec.items():
        env = get_wrapper(env, cls_name, cls_args)
    return env


def get_env(
        env_config: Mapping[str, Any], process_seed: int, mode: str
) -> Union[gym.core.Env, Wrapper]:
    """
    Args:
        env_config (Mapping[str, Any]): Mapping containing keys 'id' and 'wrappers'.
            The 'id' key should map to a string corresponding
            to the name of an OpenAI gym environment.
            The 'wrappers' key should map to a dictionary with two keys,
            'train' and 'evaluate'. Each of these will be a dictionary keyed by
            wrapper class names and with wrapper class constructor
            arguments as values.
        process_seed (int): Random number generator seed for this process.
        mode (str): A string affecting the behavior of wrappers.
            If not 'train', the 'evaluate' wrapper specification is used.

    Returns:
        Union[gym.core.Env, Wrapper]: OpenAI gym or wrapped environment.
    """
    env = gym.make(env_config.get('id'))
    env.seed(process_seed)
    wrapper_config = env_config.get('wrappers')
    if mode == 'train':
        mode_wrappers = wrapper_config.get('train')
    else:
        mode_wrappers = wrapper_config.get('evaluate')
    env = get_wrappers(env, **mode_wrappers)
    return env


def get_preprocessing(
        cls_name: str, cls_args: Mapping[str, Any]
) -> Preprocessing:
    """
    Args:
        cls_name (str): Name of a derived class of Preprocessing.
        cls_args (Mapping[str, Any]): Arguments in the signature of the class
            constructor.

    Returns:
        Preprocessing: Instantiated class.
    """
    module = importlib.import_module('drl.agents.preprocessing')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_preprocessings(
        **preprocessing_spec: Mapping[str, Mapping[str, Any]]
) -> List[Preprocessing]:
    """
    Args:
        **preprocessing_spec (Mapping[str, Mapping[str, Any]]): Variable-length
            dictionary of items. Each item is keyed by a class name, which
            should be a derived class of Preprocessing. Each value, is a
            dictionary of arguments passed to the constructor of that class.

    Returns:
        List[Preprocessing]: List of instantiated preprocessing subclasses.
    """
    preprocessing_stack = list()
    for cls_name, cls_args in preprocessing_spec.items():
        preprocessing = get_preprocessing(
            cls_name=cls_name, cls_args=cls_args)
        preprocessing_stack.append(preprocessing)
    return preprocessing_stack


def get_architecture_cls(
        cls_name: str
) -> Union[Type[StatelessArchitecture], Type[StatefulArchitecture]]:
    """
    Args:
        cls_name (str): Class name.

    Returns:
        Union[Type[StatelessArchitecture], Type[StatefulArchitecture]]:
        Architecture class object.
    """
    module = importlib.import_module('drl.agents.architectures')
    cls = getattr(module, cls_name)
    return cls


def get_architecture(
        cls_name: str,
        cls_args: Mapping[str, Any],
        w_init_spec: Tuple[str, Mapping[str, Any]],
        b_init_spec: Tuple[str, Mapping[str, Any]]
) -> Architecture:
    """
    Args:
        cls_name (str): Name of a derived class of Architecture.
        cls_args (Mapping[str, Any]): Arguments in the signature of the class
            constructor.
        w_init_spec (Tuple[str, Mapping[str, Any]]): Tuple containing weight
            initializer name and args.
        b_init_spec (Tuple[str, Mapping[str, Any]]): Tuple containing bias
            initializer name and args.

    Returns:
        Architecture: Instantiated architecture subclass.
    """
    cls = get_architecture_cls(cls_name)
    args = {
        **cls_args,
        'w_init': get_initializer(w_init_spec),
        'b_init': get_initializer(b_init_spec)
    }
    return cls(**args)


def get_predictor(
        cls_name: str,
        cls_args: Mapping[str, Any],
        head_architecture_cls_name: str,
        head_architecture_cls_args: Mapping[str, Any],
        w_init_spec: Tuple[str, Mapping[str, Any]],
        b_init_spec: Tuple[str, Mapping[str, Any]]
) -> Head:
    """
    Args:
        cls_name (str): Head class name.
        cls_args (Mapping[str, Any]): Head class constructor arguments.
            Should contain at least 'num_features'
            and either 'num_actions' or 'action_dim'.
        head_architecture_cls_name (str): Class name for head architecture.
            Should correspond to a derived class of HeadEligibleArchitecture.
        head_architecture_cls_args (Mapping[str, Any]): Class arguments for head
            architecture.
        w_init_spec (Tuple[str, Mapping[str, Any]]): Tuple containing weight
            initializer name and args.
        b_init_spec (Tuple[str, Mapping[str, Any]]): Tuple containing bias
            initializer name and args.

    Returns:
        Head: instantiated Head subclass.
    """
    assert 'num_features' in cls_args
    module = importlib.import_module('drl.agents.heads')
    head_cls = getattr(module, cls_name)
    if issubclass(head_cls, (DiscretePolicyHead, DiscreteActionValueHead)):
        assert 'num_actions' in cls_args
    if issubclass(head_cls, ContinuousPolicyHead):
        assert 'action_dim' in cls_args

    args = {
        **cls_args,
        'head_architecture_cls': get_architecture_cls(
            head_architecture_cls_name),
        'head_architecture_cls_args': head_architecture_cls_args,
        'w_init': get_initializer(w_init_spec),
        'b_init': get_initializer(b_init_spec)
    }
    return head_cls(**args)


def get_predictors(
        env: Union[gym.core.Env, Wrapper],
        **predictors_spec: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Head]:
    """
    Args:
        env (Union[gym.core.Env, Wrapper]): OpenAI gym environment instance or
            wrapped environment.
        **predictors_spec (Mapping[str, Mapping[str, Any]]): Variable-length
            dictionary of predictor keys and specs. Each spec is a dictionary,
            with keys 'cls_name' and 'cls_args'. The 'cls_name' key maps to a
            value that is the name of a derived class of `Head`.
            The 'cls_args' key maps to a dictionary of arguments to be passed to
            that class' constructor.

    Returns:
        Mapping[str, Head]: Dictionary of predictors keyed by name.
    """
    predictors = dict()
    for key, spec in predictors_spec.items():
        # infer number of actions or action dimensionality.
        if key == 'policy' or key.startswith('action_value_'):
            if isinstance(env.action_space, gym.spaces.Discrete):
                spec['cls_args'].update({'num_actions': env.action_space.n})
            elif isinstance(env.action_space, gym.spaces.Box):
                spec['cls_args'].update({'action_dim': env.action_space.shape[0]})
            else:
                msg = "Unknown action space."
                raise TypeError(msg)
        # create and add the predictor.
        predictor = get_predictor(**spec)
        predictors[key] = predictor
        # add additional predictor for epsilon-greedy policy in DQN.
        if isinstance(predictor, DiscreteActionValueHead):
            eps_greedy_policy_predictor = EpsilonGreedyCategoricalPolicyHead(
                action_value_head=predictor)
            predictors['policy'] = eps_greedy_policy_predictor
    return predictors


def get_net(
        net_config: Mapping[str, Any],
        env: Union[gym.core.Env, Wrapper]
) -> DDP:
    """
    Args:
        net_config (Mapping[str, Any]): Dictionary with three keys: 'preprocessing',
            'architecture', and 'predictors'. Each key maps to a dictionary
            conforming to the specifications in `get_preprocessings`,
            `get_architecture` and `get_predictors`.
        env (Union[gym.core.Env, Wrapper]): OpenAI gym environment or wrapper thereof.

    Returns:
        torch.nn.parallel.DistributedDataParallel: A DDP-wrapped `Agent` instance.
    """
    preprocessing = get_preprocessings(**net_config.get('preprocessing'))
    architecture = get_architecture(**net_config.get('architecture'))
    predictors = get_predictors(env, **net_config.get('predictors'))
    if not isinstance(architecture, StatelessArchitecture):
        msg = "Stateful architectures not yet supported."
        raise TypeError(msg)
    return DDP(Agent(preprocessing, architecture, predictors))


def get_opt(
        opt_config: Mapping[str, Any],
        agent: Union[DDP, Agent]
) -> Optimizer:
    optimizer = get_optimizer(model=agent, **opt_config)
    return optimizer


def get_sched(
        sched_config: Mapping[str, Any],
        optimizer: Optimizer
) -> Optional[Scheduler]:
    scheduler = get_scheduler(optimizer=optimizer, **sched_config)
    return scheduler


def make_learning_system(
        rank: int,
        config: ConfigParser
) -> Mapping[str, Union[int, Union[gym.core.Env, Wrapper], Optional[Union[DDP, Agent]], Optional[Optimizer], Optional[Scheduler]]]:
    """
    Provides a simple, flexible, and reproducible launch framework
        for the drl library.

    For use in conjunction with the configuration files provided in models_dir.

    Args:
        rank (int): Process rank.
        config (`ConfigParser`): Configuration object.

    Returns:
        Mapping[str, Union[int, Union[gym.core.Env, Wrapper], Optional[Agent], Optional[Optimizer], Optional[Scheduler]]:
            Dictionary with environment, networks, optimizers, schedulers,
            and global step of the learning process thus far.
    """
    mode = config.get('mode')

    process_seed = get_process_seed(rank, config)
    set_seed(process_seed)

    env_config = config.get('env')
    env = get_env(env_config, process_seed, mode)

    learners_config = config.get('networks')
    learning_system = dict()
    for prefix in learners_config:
        learner_config = learners_config.get(prefix)
        for suffix in learner_config:
            if suffix == 'net':
                net_config = learner_config.get(suffix)
                net = get_net(net_config, env)
                learning_system[f'{prefix}_net'] = net
            elif suffix == 'optimizer':
                opt_config = learner_config.get(suffix)
                opt = get_opt(opt_config, learning_system[f'{prefix}_net'])
                learning_system[f'{prefix}_optimizer'] = opt
            elif suffix == 'scheduler':
                sched_config = learner_config.get(suffix)
                sched = get_sched(
                    sched_config, learning_system[f'{prefix}_optimizer'])
                learning_system[f'{prefix}_scheduler'] = sched
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
