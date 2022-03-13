import warnings

import torch as tc
import gym

from drl.envs.wrappers.stateless.scale_observations import (
    ScaleObservationsWrapper)


def test_reward_to_dict():
    env = gym.make('BreakoutNoFrameskip-v4')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wrapped = ScaleObservationsWrapper(env, scale_factor=(1 / 255))
        _ = wrapped.reset()
        o_tp1, r_t, d_t, i_t = wrapped.step(0)
        obs = tc.tensor(o_tp1).float()
        assert tc.nn.ReLU()(obs - tc.ones_like(obs)).sum().item() == 0.0
        assert tc.nn.ReLU()(-obs).sum().item() == 0.0
