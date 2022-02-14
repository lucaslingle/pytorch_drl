import tempfile
import os

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
import gym

from drl.utils.launch import (
    get_process_seed,
    get_wrapper,
    get_wrappers,
    get_env,
    get_preprocessing,
    get_preprocessings,
    get_architecture_cls,
    get_architecture,
    get_predictor,
    get_predictors,
    get_net,
    make_learning_system)
from drl.envs.wrappers import (
    Wrapper, RewardToDictWrapper, ClipRewardWrapper, EpisodicLifeWrapper)
from drl.agents.preprocessing import ToChannelMajor, OneHotEncode
from drl.agents.architectures import NatureCNN, MLP
from drl.agents.heads import SimpleValueHead, CategoricalPolicyHead
from drl.utils.configuration import ConfigParser
from drl.utils.test_distributed import (
    WORLD_SIZE, make_process_group, destroy_process_group)

# for testing manually made the world_size in the config below match WORLD_SIZE.
# we can also do this with an f-string,
# but we would have to escape all the actual braces in the yaml below.
CONFIG_STR = """
distributed:
    backend: gloo
    world_size: 2
    master_addr: localhost
    master_port: '12345'
algo:
    cls_name: PPO
    cls_args:
        seg_len: 256
        opt_epochs: 3
        learner_batch_size: 64
        clip_param_init: 0.2
        clip_param_final: 0.0
        ent_coef_init: 0.01
        ent_coef_final: 0.01
        vf_loss_coef: 1.0
        vf_loss_cls: MSELoss
        vf_loss_clipping: False
        vf_simple_weighting: True
        extra_steps: 0
        standardize_adv: True
        use_pcgrad: False
        stats_window_len: 100
        checkpoint_frequency: 25600
        non_learning_steps: 0
        max_steps: 10000000
credit_assignment:
    extrinsic:
        cls_name: GAE
        cls_args: {gamma: 0.99, lambda_: 0.95, use_dones: True}
env:
    wrappers:
        train:
            AtariWrapper:
                use_noop: True
                use_sticky_actions: False
            DeepmindWrapper:
                episode_life: True
                clip_rewards: True
                frame_stack: True
                lazy: False
        evaluate:
            AtariWrapper:
                use_noop: True
                use_sticky_actions: False
            DeepmindWrapper:
                episode_life: False
                clip_rewards: False
                frame_stack: True
                lazy: False
networks:
    policy:
        net:
            preprocessing:
                ToChannelMajor: {}
            architecture:
                cls_name: NatureCNN
                cls_args: {img_channels: 4}
                w_init_spec: ['zeros_', {}]
                b_init_spec: ['zeros_', {}]
            predictors:
                policy:
                    cls_name: CategoricalPolicyHead
                    cls_args: {num_features: 512}
                    head_architecture_cls_name: Linear
                    head_architecture_cls_args: {}
                    w_init_spec: ['zeros_', {}]
                    b_init_spec: ['zeros_', {}]
                value_extrinsic:
                    cls_name: SimpleValueHead
                    cls_args: {num_features: 512}
                    head_architecture_cls_name: Linear
                    head_architecture_cls_args: {}
                    w_init_spec: ['zeros_', {}]
                    b_init_spec: ['zeros_', {}]
        optimizer:
            cls_name: Adam
            cls_args: {lr: 0.001, betas: [0.90, 0.999], eps: 0.00001}
        scheduler:
            cls_name: OneCycleLR
            cls_args:
                max_lr: 0.001
                total_steps: 4883
                pct_start: 0.0
                anneal_strategy: linear
                cycle_momentum: False
                div_factor: 1.0
    value:
        use_shared_architecture: True
""".strip("\n")


def test_get_process_seed():
    assert get_process_seed(rank=0, experiment_seed=123) == 123
    assert get_process_seed(rank=1, experiment_seed=101) == 10101
    assert get_process_seed(rank=8, experiment_seed=999) == 80999


def test_get_wrapper():
    env = RewardToDictWrapper(gym.make('BreakoutNoFrameskip-v4'))
    actual = get_wrapper(
        env=env,
        cls_name="ClipRewardWrapper",
        cls_args={
            'low': -1.0, 'high': 1.0
        })
    assert isinstance(actual, ClipRewardWrapper)


def test_get_wrappers():
    env = RewardToDictWrapper(gym.make('BreakoutNoFrameskip-v4'))
    actual = get_wrappers(
        env=env,
        wrappers_spec={
            'ClipRewardWrapper': {
                'low': -1.0, 'high': 1.0
            },
            'EpisodicLifeWrapper': {
                'lives_fn': lambda env: 1, 'noop_action': 0
            },
        })
    assert isinstance(actual, EpisodicLifeWrapper)
    assert isinstance(actual.env, ClipRewardWrapper)


def test_get_env():
    id = 'BreakoutNoFrameskip-v4'
    wrappers = {
        'train': {
            'RewardToDictWrapper': {},
            'ClipRewardWrapper': {
                'low': -1., 'high': 1.
            }
        },
        'evaluate': {
            'RewardToDictWrapper': {},
            'EpisodicLifeWrapper': {
                'lives_fn': lambda env: 1, 'noop_action': 0
            }
        }
    }
    actual_train = get_env(
        env_id=id, env_wrappers=wrappers, process_seed=123, mode='train')
    assert isinstance(actual_train, ClipRewardWrapper)
    assert isinstance(actual_train.env, RewardToDictWrapper)
    actual_evaluate = get_env(
        env_id=id, env_wrappers=wrappers, process_seed=123, mode='evaluate')
    assert isinstance(actual_evaluate, EpisodicLifeWrapper)
    assert isinstance(actual_evaluate.env, RewardToDictWrapper)


def test_get_preprocessing():
    preprocessing = get_preprocessing(cls_name='ToChannelMajor', cls_args={})
    assert isinstance(preprocessing, ToChannelMajor)
    preprocessing = get_preprocessing(
        cls_name='OneHotEncode', cls_args={'depth': 3})
    assert isinstance(preprocessing, OneHotEncode)


def test_get_preprocessings():
    expected_clss = [ToChannelMajor, OneHotEncode]
    actuals = get_preprocessings(
        preprocessings_spec={
            'ToChannelMajor': {}, 'OneHotEncode': {
                'depth': 3
            }
        })
    for actual, expected_cls in zip(actuals, expected_clss):
        assert isinstance(actual, expected_cls)


def test_get_architecture_cls():
    cls = get_architecture_cls(cls_name='NatureCNN')
    assert cls == NatureCNN
    cls = get_architecture_cls(cls_name='MLP')
    assert cls == MLP


def test_get_architecture():
    architecture = get_architecture(
        cls_name='NatureCNN',
        cls_args={'img_channels': 4},
        w_init_spec=('zeros_', {}),
        b_init_spec=('zeros_', {}))
    assert isinstance(architecture, NatureCNN)
    img_batch = tc.ones(size=(1, *architecture.input_shape), dtype=tc.float32)
    features_actual = architecture(img_batch)
    features_expected = tc.zeros(
        size=(1, architecture.output_dim), dtype=tc.float32)
    tc.testing.assert_close(actual=features_actual, expected=features_expected)


def test_get_predictor():
    predictor = get_predictor(
        cls_name='SimpleValueHead',
        cls_args={'num_features': 100},
        head_architecture_cls_name='Linear',
        head_architecture_cls_args={},
        w_init_spec=('zeros_', {}),
        b_init_spec=('zeros_', {}))
    assert isinstance(predictor, SimpleValueHead)
    feature_batch = tc.ones(size=(32, 100), dtype=tc.float32)
    vpreds_actual = predictor(feature_batch)
    vpreds_expected = tc.zeros(size=(32, ), dtype=tc.float32)
    tc.testing.assert_close(actual=vpreds_actual, expected=vpreds_expected)


def test_get_predictors():
    # yapf: disable
    predictors = get_predictors(
        env=gym.make('BreakoutNoFrameskip-v4'),
        predictors_spec={
            'policy': {
                'cls_name': 'CategoricalPolicyHead',
                'cls_args': {'num_features': 100},
                'head_architecture_cls_name': 'Linear',
                'head_architecture_cls_args': {},
                'w_init_spec': ('zeros_', {}),
                'b_init_spec': ('zeros_', {})
            },
            'value_extrinsic': {
                'cls_name': 'SimpleValueHead',
                'cls_args': {'num_features': 100},
                'head_architecture_cls_name': 'Linear',
                'head_architecture_cls_args': {},
                'w_init_spec': ('zeros_', {}),
                'b_init_spec': ('zeros_', {})
            }
        }
    )
    # yapf: enable
    for k, v in predictors.items():
        if k == 'policy':
            assert isinstance(v, CategoricalPolicyHead)
        elif k == 'value_extrinsic':
            assert isinstance(v, SimpleValueHead)


def local_test_get_net(rank, port):
    make_process_group(rank, port)

    env = gym.make('BreakoutNoFrameskip-v4')
    net = get_net(
        env=env,
        preprocessing_spec={'ToChannelMajor': {}},
        architecture_spec={
            'cls_name': 'NatureCNN',
            'cls_args': {
                'img_channels': 4
            },
            'w_init_spec': ('zeros_', {}),
            'b_init_spec': ('zeros_', {})
        },
        predictors_spec={
            'policy': {
                'cls_name': 'CategoricalPolicyHead',
                'cls_args': {
                    'num_features': 512
                },
                'head_architecture_cls_name': 'Linear',
                'head_architecture_cls_args': {},
                'w_init_spec': ('zeros_', {}),
                'b_init_spec': ('zeros_', {})
            },
            'value_extrinsic': {
                'cls_name': 'SimpleValueHead',
                'cls_args': {
                    'num_features': 512
                },
                'head_architecture_cls_name': 'Linear',
                'head_architecture_cls_args': {},
                'w_init_spec': ('zeros_', {}),
                'b_init_spec': ('zeros_', {})
            }
        })
    # yapf: enable
    img_batch = tc.ones(size=(1, 84, 84, 4), dtype=tc.float32)
    predictions = net(
        observations=img_batch, predict=['policy', 'value_extrinsic'])
    tc.testing.assert_close(
        actual=predictions['policy'].log_prob(tc.tensor([0])),
        expected=tc.log(tc.tensor([1. / env.action_space.n])))
    tc.testing.assert_close(
        actual=predictions['value_extrinsic'],
        expected=tc.zeros(size=(1, ), dtype=tc.float32))

    destroy_process_group()


def test_get_net():
    tc.multiprocessing.spawn(
        local_test_get_net, args=(13000, ), nprocs=WORLD_SIZE, join=True)


def local_test_make_learning_system(rank, port):
    make_process_group(rank, port)
    broadcast_list = [None, None]
    if rank == 0:
        config_dir = os.path.join(
            tempfile.gettempdir(), 'pytorch_drl_testing/config_dir/')
        checkpoint_dir = os.path.join(
            tempfile.gettempdir(), 'pytorch_drl_testing/launch_checkpoint_dir/')
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(CONFIG_STR)
        broadcast_list = [config_path, checkpoint_dir]
    tc.distributed.broadcast_object_list(broadcast_list, src=0)
    config_path = broadcast_list[0]
    checkpoint_dir = broadcast_list[1]

    # yapf: disable
    config = ConfigParser(
        defaults={
            'env.id': 'BreakoutNoFrameskip-v4',
            'seed': 0,
            'mode': 'train',
            'checkpoint_dir': checkpoint_dir
        }
    )
    # yapf: enable
    config.read(config_path, verbose=False)
    learning_system = make_learning_system(rank=rank, config=config)

    assert learning_system['global_step'] == 0
    assert isinstance(learning_system['env'], (gym.core.Env, Wrapper))
    assert isinstance(learning_system['policy_net'], DDP)
    assert isinstance(learning_system['policy_optimizer'], tc.optim.Adam)
    assert isinstance(
        learning_system['policy_scheduler'], tc.optim.lr_scheduler.OneCycleLR)
    assert learning_system['value_net'] is None
    assert learning_system['value_optimizer'] is None
    assert learning_system['value_scheduler'] is None

    img_batch = tc.ones(size=(1, 84, 84, 4), dtype=tc.float32)
    env = learning_system['env']
    policy_net = learning_system['policy_net']
    predictions = policy_net(
        observations=img_batch, predict=['policy', 'value_extrinsic'])
    tc.testing.assert_close(
        actual=predictions['policy'].log_prob(tc.tensor([0])),
        expected=tc.log(tc.tensor([1. / env.action_space.n])))
    tc.testing.assert_close(
        actual=predictions['value_extrinsic'],
        expected=tc.zeros(size=(1, ), dtype=tc.float32))

    destroy_process_group()


def test_make_learning_system():
    tc.multiprocessing.spawn(
        local_test_make_learning_system,
        args=(13001, ),
        nprocs=WORLD_SIZE,
        join=True)
