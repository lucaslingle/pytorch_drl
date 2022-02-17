import torch as tc
import numpy as np
import gym

from drl.envs.wrappers.stateless.frame_stack import FrameStackWrapper


def test_frame_stack():
    num_stack = 4

    env = gym.make('BreakoutNoFrameskip-v4')
    _ = env.reset()
    frames = []
    for _ in range(num_stack):
        o_tp1, r_t, d_t, i_t = env.step(0)
        frames.append(o_tp1)

    wrapped = FrameStackWrapper(env, num_stack=4, lazy=False)
    _ = wrapped.reset()
    o_tp1, r_t, d_t, i_t = wrapped.step(0)

    tc.testing.assert_close(
        actual=tc.tensor(o_tp1),
        expected=tc.tensor(np.concatenate(frames, axis=-1)))
