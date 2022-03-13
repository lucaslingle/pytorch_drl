import gym

from drl.envs.wrappers.stateless.resize_observations import (
    ResizeObservationsWrapper)


def test_noop_reset():
    env = gym.make('BreakoutNoFrameskip-v4')
    _ = env.reset()

    wrapped = ResizeObservationsWrapper(
        env, width=84, height=84, grayscale=True)
    obs = wrapped.reset()
    assert obs.shape == (84, 84, 1)

    wrapped = ResizeObservationsWrapper(
        env, width=84, height=84, grayscale=False)
    obs = wrapped.reset()
    assert obs.shape == (84, 84, 3)
