# todo: come up with better tests

from typing import Dict, Any
import tempfile
import os
import uuid
from contextlib import ExitStack

import pytest
import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.agents.preprocessing import OneHotEncode
from drl.agents.architectures import Identity, Linear
from drl.agents.integration import Agent
from drl.agents.heads import CategoricalPolicyHead, SimpleValueHead
from drl.algos.abstract import Algo, ActorCriticAlgo
from drl.algos.common import GAE, RolloutManager
from drl.envs.testing import CyclicEnv
from drl.utils.initializers import get_initializer
from drl.utils.nested import slice_nested_tensor
from drl.utils.test_distributed import (
    make_process_group, destroy_process_group, WORLD_SIZE)


# yapf: disable
def make_algo_args(rank: int) -> Dict[str, Any]:
    base_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    os.makedirs(base_dir)
    return {
        'rank': rank,
        'world_size': WORLD_SIZE,
        'rollout_len': 128,
        'extra_steps': 0,
        'credit_assignment_ops': {
            'extrinsic': GAE(lambda_=0.95, gamma=0.99, use_dones=True)
        },
        'stats_window_len': 100,
        'non_learning_steps': 0,
        'max_steps': 10000,
        'checkpoint_frequency': int,
        'checkpoint_dir': os.path.join(base_dir, 'checkpoints'),
        'log_dir': os.path.join(base_dir, 'tensorboard_logs'),
        'media_dir': os.path.join(base_dir, 'media'),
        'global_step': 0,
        'env': CyclicEnv(),
        'rollout_net': DDP(Agent(
            preprocessing=[OneHotEncode(depth=100)],
            architecture=Identity(input_shape=[100], w_init=None, b_init=None),
            predictors={
                'policy': CategoricalPolicyHead(
                    num_features=100,
                    num_actions=10,
                    head_architecture_cls=Linear,
                    head_architecture_cls_args={},
                    w_init=get_initializer(('zeros_', {})),
                    b_init=get_initializer(('zeros_', {}))),
                'value_extrinsic': SimpleValueHead(
                    num_features=100,
                    head_architecture_cls=Linear,
                    head_architecture_cls_args={},
                    w_init=get_initializer(('zeros_', {})),
                    b_init=get_initializer(('zeros_', {})))
            }
        )),
        'reward_weights': None
    }
# yapf: enable


def local_test_algo(rank: int, port: int, monkeypatch) -> None:
    make_process_group(rank, port)
    args = make_algo_args(rank)
    algo = Algo(**args)
    rollout_mgr = RolloutManager(
        env=args['env'],
        rollout_net=args['rollout_net'],
        rollout_len=args['rollout_len'],
        extra_steps=args['extra_steps'])
    rollout = rollout_mgr.generate()

    # verify that annotate is still unimplemented by algo cls
    # then mock it.
    with pytest.raises(NotImplementedError):
        algo.annotate(rollout=rollout, no_grad=True)

    def mock_algo_annotate(rollout, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            rollout = slice_nested_tensor(
                rollout, slice(0, args['rollout_len']))
            rollout.update({
                'vpreds': tc.zeros(
                    size=[args['rollout_len'] + args['extra_steps'] + 1],
                    dtype=tc.float32),
                'logprobs': tc.zeros(
                    size=[args['rollout_len']], dtype=tc.float32),
                'entropies': tc.ones(
                    size=[args['rollout_len']], dtype=tc.float32)
            })
            return rollout
    monkeypatch.setattr(algo, 'annotate', mock_algo_annotate)
    rollout = algo.annotate(rollout=rollout, no_grad=True)

    # verify that credit_assignment is still unimplemented by algo cls
    # then mock it.
    with pytest.raises(NotImplementedError):
        algo.credit_assignment(rollout=rollout)

    def mock_algo_credit_assignment(rollout):
        with tc.no_grad():
            rollout = slice_nested_tensor(
                rollout, slice(0, args['rollout_len']))
            rollout.update({
                'advantages': tc.zeros(
                    size=[args['rollout_len']], dtype=tc.float32),
                'vtargs': tc.zeros(
                    size=[args['rollout_len']], dtype=tc.float32)
            })
            return rollout
    monkeypatch.setattr(algo, 'credit_assignment', mock_algo_credit_assignment)
    rollout = algo.credit_assignment(rollout=rollout)

    # verify that compute_losses_and_metrics is still unimplemented by algo cls
    # then mock it.
    with pytest.raises(NotImplementedError):
        algo.compute_losses_and_metrics(minibatch=rollout, no_grad=True)

    def mock_algo_compute_losses_and_metrics(minibatch, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            return {
                'policy_loss': tc.tensor(1.0),
                'value_loss': tc.tensor(2.0)
            }
    monkeypatch.setattr(algo,
                        'compute_losses_and_metrics',
                        mock_algo_compute_losses_and_metrics)

    # use mocked annotate and credit_assignment to test 'collect' method...
    rollout, metadata = algo.collect()
    assert rollout['observations'].shape[0] == args['rollout_len']
    expected_rollout_keys = {
        'observations',
        'actions',
        'rewards',
        'dones',
        'logprobs',
        'entropies',
        'vpreds',
        'advantages',
        'vtargs'
    }
    assert set(rollout.keys()) == expected_rollout_keys

    # verify that optimize is still unimplemented by algo cls
    # then mock it.
    with pytest.raises(NotImplementedError):
        algo.optimize(rollout=rollout)

    def mock_algo_optimize(rollout):
        return None
    monkeypatch.setattr(algo, 'optimize', mock_algo_optimize)

    # verify that persist is still unimplemented by algo cls
    # then mock it.
    def mock_algo_persist(metadata):
        return None
    monkeypatch.setattr(algo, 'persist', mock_algo_persist)

    # use mocked annotate, credit_assignment (used by collect),
    # optimize and persist to test 'train_loop' method...
    algo.training_loop()
    assert algo._global_step >= args['max_steps']

    destroy_process_group()


def test_algo(monkeypatch) -> None:
    port = 20000
    tc.multiprocessing.spawn(
        local_test_algo, args=(port, monkeypatch), nprocs=WORLD_SIZE, join=True)


def make_actor_critic_args(rank: int) -> Dict[str, Any]:
    basic_algo_args = make_algo_args(rank)
    basic_algo_args.update({
        'policy_net': basic_algo_args['rollout_net'],
        'policy_optimizer': tc.optim.Adam(
            basic_algo_args['rollout_net'].parameters(), lr=1e-3),
        'policy_scheduler': None,
        'value_net': None,
        'value_optimizer': None,
        'value_scheduler': None,
        'standardize_adv': True
    })
    return basic_algo_args


def local_test_actor_critic_algo(rank: int, port: int, monkeypatch) -> None:
    make_process_group(rank, port)
    args = make_actor_critic_args(rank)
    rollout_net = args.pop('rollout_net')
    rollout_len, extra_steps = args.get('rollout_len'), args.get('extra_steps')
    algo = ActorCriticAlgo(**args)
    rollout_mgr = RolloutManager(
        env=args['env'],
        rollout_net=rollout_net,
        rollout_len=args['rollout_len'],
        extra_steps=args['extra_steps'])
    rollout = rollout_mgr.generate()
    assert rollout['observations'].shape[0] == rollout_len + extra_steps + 1
    assert rollout['actions'].shape[0] == rollout_len + extra_steps + 1
    for k in algo._get_reward_keys(omit_raw=True):
        assert rollout['rewards'][k].shape[0] == rollout_len + extra_steps
    assert rollout['dones'].shape[0] == rollout_len + extra_steps

    rollout = algo.annotate(rollout=rollout, no_grad=True)
    assert set(rollout.keys()) == {
        'observations',
        'actions',
        'rewards',
        'dones',
        'logprobs',
        'entropies',
        'vpreds'
    }
    assert rollout['observations'].shape[0] == rollout_len
    assert rollout['actions'].shape[0] == rollout_len
    for k in algo._get_reward_keys(omit_raw=True):
        assert rollout['rewards'][k].shape[0] == rollout_len + extra_steps
    assert rollout['dones'].shape[0] == rollout_len + extra_steps
    assert rollout['logprobs'].shape[0] == rollout_len
    assert rollout['entropies'].shape[0] == rollout_len
    for k in algo._get_reward_keys(omit_raw=True):
        assert rollout['vpreds'][k].shape[0] == rollout_len + extra_steps + 1

    rollout = algo.credit_assignment(rollout=rollout)
    assert set(rollout.keys()) == {
        'observations',
        'actions',
        'rewards',
        'dones',
        'logprobs',
        'entropies',
        'vpreds',
        'advantages',
        'vtargs'
    }

    standalone_policy_net = DDP(Agent(
        preprocessing=[OneHotEncode(depth=100)],
        architecture=Identity(input_shape=[100], w_init=None, b_init=None),
        predictors={
            'policy': CategoricalPolicyHead(
                num_features=100,
                num_actions=10,
                head_architecture_cls=Linear,
                head_architecture_cls_args={},
                w_init=get_initializer(('zeros_', {})),
                b_init=get_initializer(('zeros_', {})))
        }
    ))
    standalone_value_net = DDP(Agent(
        preprocessing=[OneHotEncode(depth=100)],
        architecture=Identity(input_shape=[100], w_init=None, b_init=None),
        predictors={
            'value_extrinsic': SimpleValueHead(
                num_features=100,
                head_architecture_cls=Linear,
                head_architecture_cls_args={},
                w_init=get_initializer(('zeros_', {})),
                b_init=get_initializer(('zeros_', {})))
        }
    ))
    args['policy_net'] = standalone_policy_net
    args['value_net'] = standalone_value_net
    algo = ActorCriticAlgo(**args)
    rollout = rollout_mgr.generate()
    rollout = algo.annotate(rollout=rollout, no_grad=True)
    assert set(rollout.keys()) == {
        'observations',
        'actions',
        'rewards',
        'dones',
        'logprobs',
        'entropies',
        'vpreds'
    }
    rollout = algo.credit_assignment(rollout=rollout)
    assert set(rollout.keys()) == {
        'observations',
        'actions',
        'rewards',
        'dones',
        'logprobs',
        'entropies',
        'vpreds',
        'advantages',
        'vtargs'
    }

    # verify that optimize is still unimplemented by algo cls
    # then mock it.
    with pytest.raises(NotImplementedError):
        algo.optimize(rollout=rollout)

    def mock_algo_optimize(rollout):
        return None
    monkeypatch.setattr(algo, 'optimize', mock_algo_optimize)

    # verify that persist is still unimplemented by algo cls
    # then mock it.
    def mock_algo_persist(metadata):
        return None
    monkeypatch.setattr(algo, 'persist', mock_algo_persist)

    # use mocked annotate, credit_assignment (used by collect),
    # optimize and persist to test 'train_loop' method...
    algo.training_loop()
    assert algo._global_step >= args['max_steps']

    destroy_process_group()


def test_actor_critic_algo(monkeypatch) -> None:
    port = 20001
    tc.multiprocessing.spawn(
        local_test_actor_critic_algo,
        args=(port, monkeypatch),
        nprocs=WORLD_SIZE,
        join=True)
