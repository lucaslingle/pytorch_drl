from typing import Mapping, Any
import torch as tc

from drl.utils.test_distributed import requires_process_group, WORLD_SIZE
from drl.envs.wrappers.stateful.normalize_reward import (
    ReturnAcc, NormalizeRewardWrapper)

# todo: add tests for other methods of ReturnAcc and for NormalizeRewardWrapper.


def test_return_accumulator_terminal():
    acc = ReturnAcc(gamma=0.9, clip_low=None, clip_high=None, use_dones=True)
    acc.trace_length = 4
    acc.update(r_t=1.0, d_t=False)
    acc.update(r_t=2.0, d_t=False)
    acc.update(r_t=3.0, d_t=False)
    acc.update(r_t=4.0, d_t=True)
    acc.update(r_t=0.0, d_t=False)
    acc.update(r_t=0.0, d_t=False)
    acc.update(r_t=0.0, d_t=False)
    acc.update(r_t=0.0, d_t=False)
    # yapf: disable
    actual_m1 = acc.moment1
    expected_m1 = 0.25 * (
            (1.0 + 0.9 * 2.0 + (0.9**2) * 3.0 + (0.9**3) * 4.0) +
            (2.0 + 0.9 * 3.0 + (0.9**2) * 4.0) +
            (3.0 + 0.9 * 4.0) +
            4.0
    )
    # yapf: enable
    assert actual_m1 == expected_m1
