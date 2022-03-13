import torch as tc
import numpy as np
import gym

from drl.envs.wrappers.stateless.max_and_skip import MaxAndSkipWrapper


def test_max_and_skip():
    num_skip = 4

    env = gym.make('BreakoutNoFrameskip-v4')
    _ = env.reset()
    frames = []
    for _ in range(num_skip):
        o_tp1, r_t, d_t, i_t = env.step(0)
        frames.append(o_tp1)

    wrapped = MaxAndSkipWrapper(
        env, num_skip=num_skip, apply_maxpool=False, depth_maxpool=42)
    _ = wrapped.reset()
    o_tp1, r_t, d_t, i_t = wrapped.step(0)
    tc.testing.assert_close(
        actual=tc.tensor(o_tp1), expected=tc.tensor(frames[-1]))

    wrapped1 = MaxAndSkipWrapper(
        env, num_skip=num_skip, apply_maxpool=True, depth_maxpool=2)
    _ = wrapped1.reset()
    o_tp1, r_t, d_t, i_t = wrapped1.step(0)
    tc.testing.assert_close(
        actual=tc.tensor(o_tp1),
        expected=tc.tensor(np.maximum(frames[-2], frames[-1])))

    wrapped2 = MaxAndSkipWrapper(
        env, num_skip=num_skip, apply_maxpool=True, depth_maxpool=3)
    _ = wrapped.reset()
    o_tp1, r_t, d_t, i_t = wrapped2.step(0)
    tc.testing.assert_close(
        actual=tc.tensor(o_tp1),
        expected=tc.tensor(np.maximum(frames[-3], frames[-2], frames[-1])))
