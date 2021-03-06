"""
Optional launch utility -
    for use with provided script and specified config file format.
"""

from typing import List, Any, Mapping, Union, Type, Tuple
import random
import importlib

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import gym

from drl.utils.configuration import ConfigParser
from drl.envs.wrappers import Wrapper
from drl.agents.preprocessing import Preprocessing
from drl.agents.architectures import (
    Architecture, StatelessArchitecture, StatefulArchitecture)
from drl.utils.initializers import get_initializer
from drl.agents.heads import (
    Head,
    DiscreteActionValueHead,
    EpsilonGreedyCategoricalPolicyHead,
    DiscretePolicyHead,
    ContinuousPolicyHead)
from drl.agents.integration.agent import Agent
from drl.algos.common import get_credit_assignment_ops
from drl.utils.optimization import get_optimizer, get_scheduler
from drl.utils.checkpointing import maybe_load_checkpoints


def get_process_seed(rank: int, experiment_seed: int) -> int:
    """
    Computes a process-specific RNG seed to ensure experiential diversity
    and per-machine experimental reproducibility.

    Args:
        rank (int): Process rank.
        experiment_seed (int): Experiment seed.

    Returns:
        int: Process seed.
    """
    return 10000 * rank + experiment_seed


def set_seed(process_seed: int) -> None:
    """
    Sets the random number generators on this process to the provided RNG seed.

    Args:
        process_seed (int): RNG seed integer.

    Returns:
        None.
    """
    tc.manual_seed(process_seed)
    np.random.seed(process_seed % 2**32)
    random.seed(process_seed % 2**32)


def get_wrapper(
        env: Union[gym.core.Env, Wrapper],
        cls_name: str,
        cls_args: Mapping[str, Any]) -> Wrapper:
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
        wrappers_spec: Mapping[str, Mapping[str, Any]]) -> Wrapper:
    """
    Args:
        env (Union[gym.core.Env, `Wrapper`]): OpenAI gym environment or
            `Wrapper` thereof.
        wrappers_spec (Mapping[str, Mapping[str, Any]]): Dictionary of all
            wrappers to apply, in order. This requires Python 3.6+ to work
            correctly.

    Returns:
        Union[gym.core.Env, Wrapper]: Wrapped environment.
    """
    for cls_name, cls_args in wrappers_spec.items():
        env = get_wrapper(env, cls_name, cls_args)
    return env


def get_env(
        env_id: str,
        env_wrappers: Mapping[str, Mapping[str, Any]],
        process_seed: int,
        mode: str) -> Union[gym.core.Env, Wrapper]:
    """
    Args:
        env_id (str): OpenAI gym environment name.
        env_wrappers (Mapping[str, Mapping[str, Any]]): Dictionary mapping from
            'train' and 'evaluate' to dictionaries conforming to the
            specification of `wrappers_spec` in the get_wrappers docstring.
        process_seed (int): Random number generator seed for this process.
        mode (str): A string affecting the behavior of wrappers.
            If not 'train', the 'evaluate' wrapper specification is used.

    Returns:
        Union[gym.core.Env, Wrapper]: OpenAI gym or wrapped environment.
    """
    env = gym.make(env_id)
    env.seed(process_seed)
    if mode == 'train':
        mode_wrappers = env_wrappers.get('train')
    else:
        mode_wrappers = env_wrappers.get('evaluate')
    env = get_wrappers(env=env, wrappers_spec=mode_wrappers)
    return env


def get_preprocessing(
        cls_name: str, cls_args: Mapping[str, Any]) -> Preprocessing:
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
    preprocessings_spec: Mapping[str, Mapping[str,
                                              Any]]) -> List[Preprocessing]:
    """
    Args:
        preprocessings_spec (Mapping[str, Mapping[str, Any]]): Variable-length
            dictionary of items. Each item is keyed by a class name, which
            should be a derived class of Preprocessing. Each value, is a
            dictionary of arguments passed to the constructor of that class.

    Returns:
        List[Preprocessing]: List of instantiated preprocessing subclasses.
    """
    preprocessing_stack = list()
    for cls_name, cls_args in preprocessings_spec.items():
        preprocessing = get_preprocessing(cls_name=cls_name, cls_args=cls_args)
        preprocessing_stack.append(preprocessing)
    return preprocessing_stack


def get_architecture_cls(cls_name: str) -> Union[Architecture]:
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
        b_init_spec: Tuple[str, Mapping[str, Any]]) -> Architecture:
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
        b_init_spec: Tuple[str, Mapping[str, Any]]) -> Head:
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
        'head_architecture_cls':
            get_architecture_cls(head_architecture_cls_name),
        'head_architecture_cls_args': head_architecture_cls_args,
        'w_init': get_initializer(w_init_spec),
        'b_init': get_initializer(b_init_spec)
    }
    return head_cls(**args)


def get_predictors(
        env: Union[gym.core.Env, Wrapper],
        predictors_spec: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Head]:
    """
    Args:
        env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper thereof.
        predictors_spec (Mapping[str, Mapping[str, Any]]): Dictionary mapping
            from prediction key to dictionary of arguments conforming to the
            signature of the `get_predictor` function.

    Returns:
        Mapping[str, Head]: Dictionary mapping from prediction keys to heads.
    """
    predictors = dict()
    for key, spec in predictors_spec.items():
        # infer number of actions or action dimensionality.
        if key == 'policy' or key.startswith('action_value_'):
            if isinstance(env.action_space, gym.spaces.Discrete):
                spec['cls_args'].update({'num_actions': env.action_space.n})
            elif isinstance(env.action_space, gym.spaces.Box):
                spec['cls_args'].update(
                    {'action_dim': env.action_space.shape[0]})
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
        env: Union[gym.core.Env, Wrapper],
        preprocessing_spec: Mapping[str, Any],
        architecture_spec: Mapping[str, Any],
        predictors_spec: Mapping[str, Any]) -> DDP:
    """
    Args:
        env (Union[gym.core.Env, Wrapper]): OpenAI gym env or wrapper thereof.
        preprocessing_spec (Mapping[str, Any]): Dictionary mapping conforming to
            the specification described in `get_preprocessings` docstring.
        architecture_spec (Mapping[str, Any]): Dictionary mapping containing
            the args for `get_architecture`.
        predictors_spec (Mapping[str, Any]): Dictionary mapping conforming to
            the specification described in `get_predictors` docstring.

    Returns:
        torch.nn.parallel.DistributedDataParallel: A DDP-wrapped Agent instance.
    """
    preprocessing = get_preprocessings(preprocessing_spec)
    architecture = get_architecture(**architecture_spec)
    predictors = get_predictors(env, predictors_spec)
    if not isinstance(architecture, StatelessArchitecture):
        msg = "architecture must be instance of StatelessArchitecture."
        raise TypeError(msg)
    return DDP(Agent(preprocessing, architecture, predictors))


def make_learning_system(rank: int, config: ConfigParser) -> Mapping[str, Any]:
    """
    Provides a simple, flexible, and reproducible launch framework
        for the drl library.

    For use in conjunction with the configuration files provided in models_dir.

    Args:
        rank (int): Process rank.
        config (ConfigParser): Configuration object.

    Returns:
        Mapping[str, Any]: Dictionary with environment, networks, optimizers,
            schedulers, and global step of the learning process thus far.
    """
    mode = config.get('mode')

    experiment_seed = config.get('seed')
    process_seed = get_process_seed(rank=rank, experiment_seed=experiment_seed)
    set_seed(process_seed)

    env_config = config.get('env')
    env = get_env(
        env_id=env_config.get('id'),
        env_wrappers=env_config.get('wrappers'),
        process_seed=process_seed,
        mode=mode)

    credit_assignment_spec = config.get('credit_assignment')
    credit_assignment_ops = get_credit_assignment_ops(credit_assignment_spec)

    learners_config = config.get('networks')
    learning_system = dict()
    for prefix in learners_config:
        learner_config = learners_config.get(prefix)
        for suffix in learner_config:
            if suffix == 'net':
                net_config = learner_config.get(suffix)
                learning_system[f'{prefix}_net'] = get_net(
                    env=env,
                    preprocessing_spec=net_config.get('preprocessing'),
                    architecture_spec=net_config.get('architecture'),
                    predictors_spec=net_config.get('predictors'))
            elif suffix == 'optimizer':
                optimizer_config = learner_config.get(suffix)
                learning_system[f'{prefix}_optimizer'] = get_optimizer(
                    model=learning_system[f'{prefix}_net'],
                    cls_name=optimizer_config.get('cls_name'),
                    cls_args=optimizer_config.get('cls_args'))
            elif suffix == 'scheduler':
                scheduler_config = learner_config.get(suffix)
                learning_system[f'{prefix}_scheduler'] = get_scheduler(
                    optimizer=learning_system[f'{prefix}_optimizer'],
                    cls_name=scheduler_config.get('cls_name'),
                    cls_args=scheduler_config.get('cls_args'))
            elif suffix == 'use_shared_architecture':
                learning_system[f'{prefix}_net'] = None
                learning_system[f'{prefix}_optimizer'] = None
                learning_system[f'{prefix}_scheduler'] = None
                break
            else:
                msg = f'Unrecognized suffix {suffix} in config file.'
                raise ValueError(msg)

    checkpoint_dict = {k: v for k, v in learning_system.items()}
    checkpoint_dict.update(env.checkpointables)
    global_step = maybe_load_checkpoints(
        checkpoint_dir=config.get('checkpoint_dir'),
        checkpointables=checkpoint_dict,
        map_location='cpu',
        steps=None)

    return {
        'global_step': global_step,
        'env': env,
        'credit_assignment_ops': credit_assignment_ops,
        **learning_system
    }
