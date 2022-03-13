from collections import Counter

import torch as tc

from drl.envs.testing import CyclicEnv
from drl.envs.testing import DummyTrainableWrapper
from drl.envs.wrappers import ClipRewardWrapper
from drl.algos.common.wrapper_ops import update_trainable_wrappers


def make_wrapped_env():
    env = CyclicEnv()
    env = DummyTrainableWrapper(env)
    env = ClipRewardWrapper(env, -1., 1.)
    return env


def make_mb():
    return {
        'observations': tc.arange(2),
        'actions': tc.zeros(2, dtype=tc.int64),
        'rewards': {
            'extrinsic': tc.ones(2, dtype=tc.float32),
            'extrinsic_raw': tc.ones(2, dtype=tc.float32),
        },
        'dones': tc.zeros(2, dtype=tc.float32),
    }


def test_update_trainable_wrappers():
    wrapped_env = make_wrapped_env()
    mb = make_mb()
    update_trainable_wrappers(wrapped_env, mb)
    assert wrapped_env.env.counts == Counter({'0': 1, '1': 1})
    update_trainable_wrappers(wrapped_env, mb)
    assert wrapped_env.env.counts == Counter({'0': 2, '1': 2})
