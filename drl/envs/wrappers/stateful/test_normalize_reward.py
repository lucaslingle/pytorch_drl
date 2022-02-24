from typing import Mapping, Any, Tuple

import torch as tc

from drl.envs.wrappers.stateful.normalize_reward import (
    ReturnAcc, NormalizeRewardWrapper)
from drl.envs.wrappers.stateless.reward_to_dict import RewardToDictWrapper
from drl.envs.testing.lockstep import LockstepEnv
from drl.utils.test_distributed import requires_process_group, WORLD_SIZE


def test_return_accumulator_getters_setters() -> None:
    gamma = 0.9
    acc = ReturnAcc(gamma=gamma, clip_low=None, clip_high=None, use_dones=True)

    tc.testing.assert_close(acc.steps, tc.tensor(0.))
    acc.steps = tc.tensor(1.)
    tc.testing.assert_close(acc.steps, tc.tensor(1.))
    acc.steps = tc.tensor(0.)
    tc.testing.assert_close(acc.steps, tc.tensor(0.))

    tc.testing.assert_close(
        actual=acc.moment1, expected=tc.zeros(size=[], dtype=tc.float32))
    acc.moment1 = tc.ones(size=[], dtype=tc.float32)
    tc.testing.assert_close(
        actual=acc.moment1, expected=tc.ones(size=[], dtype=tc.float32))
    acc.moment1 = tc.zeros(size=[], dtype=tc.float32)
    tc.testing.assert_close(
        actual=acc.moment1, expected=tc.zeros(size=[], dtype=tc.float32))

    tc.testing.assert_close(
        actual=acc.moment2, expected=tc.zeros(size=[], dtype=tc.float32))
    acc.moment2 = tc.ones(size=[], dtype=tc.float32)
    tc.testing.assert_close(
        actual=acc.moment2, expected=tc.ones(size=[], dtype=tc.float32))
    acc.moment2 = tc.zeros(size=[], dtype=tc.float32)
    tc.testing.assert_close(
        actual=acc.moment2, expected=tc.zeros(size=[], dtype=tc.float32))

    assert acc.trace_length == int(5 / (1 - gamma))
    acc.trace_length = 4
    assert acc.trace_length == 4


def make_return_acc() -> ReturnAcc:
    acc = ReturnAcc(gamma=0.9, clip_low=None, clip_high=None, use_dones=True)
    acc.trace_length = 4
    return acc


def update_return_acc(acc: ReturnAcc) -> None:
    # todo: make version that does and doesn't use d_t = True ever.
    #  run tests that give the same results regardless of whether done is used,
    #  etc.
    acc.update(r_t=1.0, d_t=False)
    acc.update(r_t=2.0, d_t=False)
    acc.update(r_t=3.0, d_t=False)
    acc.update(r_t=4.0, d_t=True)
    acc.update(r_t=0.0, d_t=False)
    acc.update(r_t=0.0, d_t=False)
    acc.update(r_t=0.0, d_t=False)
    acc.update(r_t=0.0, d_t=False)


def get_expected_moments() -> Tuple[float, float]:
    # yapf: disable
    expected_m1 = 0.25 * (
            (1.0 + 0.9 * 2.0 + (0.9**2) * 3.0 + (0.9**3) * 4.0) +
            (2.0 + 0.9 * 3.0 + (0.9**2) * 4.0) +
            (3.0 + 0.9 * 4.0) +
            4.0
    )
    expected_m2 = 0.25 * (
            (1.0 + 0.9 * 2.0 + (0.9**2) * 3.0 + (0.9**3) * 4.0) ** 2 +
            (2.0 + 0.9 * 3.0 + (0.9**2) * 4.0) ** 2 +
            (3.0 + 0.9 * 4.0) ** 2 +
            4.0 ** 2
    )
    # yapf: enable
    return expected_m1, expected_m2


def test_return_accumulator_terminal() -> None:
    acc = make_return_acc()
    update_return_acc(acc)
    assert acc.steps == 4
    actual_m1, actual_m2 = acc.moment1, acc.moment2
    expected_m1, expected_m2 = get_expected_moments()
    tc.testing.assert_close(actual=actual_m1, expected=tc.tensor(expected_m1))
    tc.testing.assert_close(actual=actual_m2, expected=tc.tensor(expected_m2))


def test_return_accumulator_forward() -> None:
    acc = make_return_acc()

    input_reward = 0.
    assert acc(input_reward) == 0.

    update_return_acc(acc)
    assert acc.steps == 4

    actual_normalized = acc(input_reward, shift=True, scale=True)
    expected_m1, expected_m2 = get_expected_moments()
    numer = input_reward - expected_m1
    denom = (expected_m2 - expected_m1**2)**0.5
    expected_normalized = numer / denom
    tc.testing.assert_close(
        actual=tc.tensor(actual_normalized),
        expected=tc.tensor(expected_normalized))

    actual_normalized = acc(input_reward, shift=False, scale=True)
    numer = input_reward
    denom = (expected_m2 - expected_m1**2)**0.5
    expected_normalized = numer / denom
    tc.testing.assert_close(
        actual=tc.tensor(actual_normalized),
        expected=tc.tensor(expected_normalized))


@requires_process_group
def local_test_normalize_reward_wrapper_step(
        **kwargs: Mapping[str, Any]) -> None:
    wrapper = NormalizeRewardWrapper(
        env=LockstepEnv(),
        gamma=0.9,
        world_size=WORLD_SIZE,
        use_dones=True,
        key='extrinsic')
    wrapper.trace_length = 4
    for i in range(4):
        _, r_t, _, _ = wrapper.step(i)
        assert r_t['extrinsic_raw'] == float(i + 1)
    for _ in range(4):
        _, r_t, _, _ = wrapper.step(42)
        assert r_t['extrinsic_raw'] == 0.
    wrapper.learn(minibatch={})

    _, r_t, _, _ = wrapper.step(4)
    assert r_t['extrinsic_raw'] == float(4 + 1)

    assert wrapper._synced_normalizer.steps.item() == 4
    expected_m1, expected_m2 = map(
        lambda x: tc.tensor(x), get_expected_moments())
    tc.testing.assert_close(
        actual=wrapper._synced_normalizer.moment1, expected=expected_m1)
    tc.testing.assert_close(
        actual=wrapper._synced_normalizer.moment2, expected=expected_m2)

    actual = tc.tensor(r_t['extrinsic'])
    expected = tc.tensor(5) / tc.sqrt(expected_m2 - tc.square(expected_m1))
    tc.testing.assert_close(actual=actual, expected=expected)


def test_normalize_reward_wrapper_step() -> None:
    tc.multiprocessing.spawn(
        local_test_normalize_reward_wrapper_step,
        args=(16000, ),
        nprocs=WORLD_SIZE,
        join=True)


def test_normalize_reward_wrapper_checkpointables() -> None:
    wrapper = NormalizeRewardWrapper(
        env=RewardToDictWrapper(LockstepEnv()),
        gamma=0.9,
        world_size=WORLD_SIZE,
        use_dones=True,
        key='extrinsic')
    checkpointables = wrapper.checkpointables
    assert len(checkpointables) == 1
    assert isinstance(checkpointables['reward_normalizer'], ReturnAcc)
