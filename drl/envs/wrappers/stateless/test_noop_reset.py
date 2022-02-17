import gym

from drl.envs.testing import CounterWrapper
from drl.envs.wrappers.stateless.noop_reset import NoopResetWrapper


def test_noop_reset():
    env = CounterWrapper(gym.make('BreakoutNoFrameskip-v4'))
    _ = env.reset()

    wrapped = NoopResetWrapper(env, noop_action=0, noop_min=1, noop_max=10)
    for _ in range(100):
        _ = wrapped.reset(reset_counts=True)
        assert 1 <= wrapped.env.counts[0] <= 10
