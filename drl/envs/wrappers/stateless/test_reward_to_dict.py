from typing import Mapping, Any
import torch as tc
import gym

from drl.envs.wrappers.stateless.reward_to_dict import RewardToDictWrapper
from drl.envs.wrappers.stateless.resize_observations import (
    ResizeObservationsWrapper)
from drl.envs.wrappers.stateless.scale_observations import (
    ScaleObservationsWrapper)
from drl.envs.wrappers.stateful.intrinsic.random_network_distillation import (
    RandomNetworkDistillationWrapper)
from drl.utils.test_distributed import requires_process_group, WORLD_SIZE


@requires_process_group
def local_test_reward_to_dict(**kwargs: Mapping[str, Any]) -> None:
    # w/o intrinsic
    env = gym.make('BreakoutNoFrameskip-v4')
    wrapped = RewardToDictWrapper(env)
    _ = wrapped.reset()
    o_tp1, r_t, d_t, i_t = wrapped.step(0)
    assert sorted(r_t.keys()) == ['extrinsic', 'extrinsic_raw']

    # w/ intrinsic
    # yapf: disable
    wrapped2 = RandomNetworkDistillationWrapper(
        env=ScaleObservationsWrapper(
            env=ResizeObservationsWrapper(
                env=RewardToDictWrapper(env),
                height=84,
                width=84,
                grayscale=True),
            scale_factor=(1/255)),
        rnd_optimizer_cls_name='Adam',
        rnd_optimizer_args={},
        world_size=WORLD_SIZE,
        widening=1,
        non_learning_steps=128)
    # yapf: enable
    wrapped2.reset()
    o_tp1, r_t, d_t, i_t = wrapped2.step(0)
    assert sorted(r_t.keys()) == ['extrinsic', 'extrinsic_raw', 'intrinsic_rnd']


def test_reward_to_dict() -> None:
    tc.multiprocessing.spawn(
        local_test_reward_to_dict, args=(14000, ), nprocs=WORLD_SIZE, join=True)
