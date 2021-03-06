import torch as tc
import gym

from drl.algos.common.rollout import (
    torch_dtype, MetadataManager, Rollout, RolloutManager)
from drl.envs.wrappers import RewardToDictWrapper
from drl.envs.testing import CyclicEnv
from drl.agents.preprocessing import OneHotEncode
from drl.agents.architectures import Identity, Linear
from drl.agents.integration import Agent
from drl.agents.heads import CategoricalPolicyHead
from drl.utils.initializers import get_initializer


def test_torch_dtype():
    assert torch_dtype('uint8') == tc.uint8
    assert torch_dtype('int32') == tc.int32
    assert torch_dtype('int64') == tc.int64
    assert torch_dtype('float16') == tc.float16
    assert torch_dtype('float32') == tc.float32
    assert torch_dtype('float64') == tc.float64


def make_metadata_mgr():
    return MetadataManager(
        defaults={
            'ep_len': 0, 'ep_ret': 0.0, 'ep_len_raw': 0, 'ep_ret_raw': 0.0
        })


def test_metadata_mgr_keys():
    mm = make_metadata_mgr()
    assert set(mm.keys) == {'ep_len', 'ep_ret', 'ep_len_raw', 'ep_ret_raw'}


def test_metadata_mgr_present():
    mm = make_metadata_mgr()
    assert mm.present == {
        'ep_len': 0, 'ep_ret': 0.0, 'ep_len_raw': 0, 'ep_ret_raw': 0.0
    }


def test_metadata_mgr_pasts():
    mm = make_metadata_mgr()
    pasts = mm.pasts
    for key in pasts:
        assert pasts[key] == []


def test_metadata_mgr_update_present():
    mm = make_metadata_mgr()
    deltas = {'ep_len': 1, 'ep_ret': 2.0, 'ep_len_raw': 3, 'ep_ret_raw': 4.0}
    mm.update_present(deltas)
    assert mm.present == deltas


def test_metadata_mgr_present_done():
    mm = make_metadata_mgr()
    deltas = {'ep_len': 1, 'ep_ret': 2.0, 'ep_len_raw': 3, 'ep_ret_raw': 4.0}
    mm.update_present(deltas)
    mm.present_done('ep_len', 'ep_ret', 'ep_len_raw', 'ep_ret_raw')
    assert mm.present == {
        'ep_len': 0, 'ep_ret': 0.0, 'ep_len_raw': 0, 'ep_ret_raw': 0.0
    }
    assert mm.pasts == {
        'ep_len': [1], 'ep_ret': [2.0], 'ep_len_raw': [3], 'ep_ret_raw': [4.0]
    }
    deltas2 = {'ep_len': 5, 'ep_ret': 6.0, 'ep_len_raw': 7, 'ep_ret_raw': 8.0}
    mm.update_present(deltas2)
    assert mm.present == deltas2
    mm.present_done('ep_len', 'ep_ret', 'ep_len_raw', 'ep_ret_raw')
    assert mm.present == {
        'ep_len': 0, 'ep_ret': 0.0, 'ep_len_raw': 0, 'ep_ret_raw': 0.0
    }
    assert mm.pasts == {
        'ep_len': [1, 5],
        'ep_ret': [2.0, 6.0],
        'ep_len_raw': [3, 7],
        'ep_ret_raw': [4.0, 8.0]
    }


def test_metadata_mgr_past_done():
    mm = make_metadata_mgr()
    deltas = {'ep_len': 1, 'ep_ret': 2.0, 'ep_len_raw': 3, 'ep_ret_raw': 4.0}
    mm.update_present(deltas)
    mm.present_done('ep_len', 'ep_ret', 'ep_len_raw', 'ep_ret_raw')
    assert mm.present == {
        'ep_len': 0, 'ep_ret': 0.0, 'ep_len_raw': 0, 'ep_ret_raw': 0.0
    }
    assert mm.pasts == {
        'ep_len': [1], 'ep_ret': [2.0], 'ep_len_raw': [3], 'ep_ret_raw': [4.0]
    }
    deltas2 = {'ep_len': 5, 'ep_ret': 6.0, 'ep_len_raw': 7, 'ep_ret_raw': 8.0}
    mm.update_present(deltas2)
    mm.past_done()
    assert mm.present == deltas2
    assert mm.pasts == {
        'ep_len': [], 'ep_ret': [], 'ep_len_raw': [], 'ep_ret_raw': []
    }


def make_env(wrap=True):
    env = gym.make('BreakoutNoFrameskip-v4')
    if wrap:
        env = RewardToDictWrapper(env)
    return env


ROLLOUT_LEN = 7
EXTRA_STEPS = 2


def make_rollout(env):
    return Rollout(
        obs_space=env.observation_space,
        ac_space=env.action_space,
        rew_keys=env.reward_spec.keys,
        rollout_len=ROLLOUT_LEN,
        extra_steps=EXTRA_STEPS)


def test_rollout_record():
    env = make_env(wrap=True)
    traj = make_rollout(env)
    o_t = env.reset()
    a_t = tc.randint(env.action_space.n, size=(1, )).item()
    o_tp1, r_t, d_t, i_t = env.step(a_t)
    traj.record(t=0, o_t=o_t, a_t=a_t, r_t=r_t, d_t=d_t)
    tc.testing.assert_close(
        actual=traj._observations[0], expected=tc.tensor(o_t, dtype=tc.uint8))
    tc.testing.assert_close(
        actual=traj._actions[0], expected=tc.tensor(a_t, dtype=tc.int64))
    for k in env.reward_spec.keys:
        tc.testing.assert_close(
            actual=traj._rewards[k][0],
            expected=tc.tensor(r_t[k], dtype=tc.float32))
    tc.testing.assert_close(
        actual=traj._dones[0], expected=tc.tensor(d_t, dtype=tc.float32))


def test_rollout_report():
    env = CyclicEnv()
    traj = make_rollout(env)
    o_t = env.reset()

    for t in range(ROLLOUT_LEN + EXTRA_STEPS):
        a_t = 0
        o_tp1, r_t, d_t, i_t = env.step(a_t)
        traj.record(t=t, o_t=o_t, a_t=a_t, r_t=r_t, d_t=d_t)
        o_t = o_tp1
    traj.record_nexts(o_Tp1=o_t, a_Tp1=0)
    report = traj.report()
    tc.testing.assert_close(
        actual=report['observations'],
        expected=tc.arange(ROLLOUT_LEN + EXTRA_STEPS + 1))

    for t in range(EXTRA_STEPS, ROLLOUT_LEN + EXTRA_STEPS):
        a_t = 0
        o_tp1, r_t, d_t, i_t = env.step(a_t)
        traj.record(t=t, o_t=o_t, a_t=a_t, r_t=r_t, d_t=d_t)
        o_t = o_tp1
    traj.record_nexts(o_Tp1=o_t, a_Tp1=0)
    report = traj.report()
    tc.testing.assert_close(
        actual=report['observations'],
        expected=tc.tensor([7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))


def make_rollout_mgr():
    return RolloutManager(
        env=CyclicEnv(),
        rollout_net=Agent(
            preprocessing=[OneHotEncode(depth=100)],
            architecture=Identity(input_shape=[100], w_init=None, b_init=None),
            predictors={
                'policy':
                    CategoricalPolicyHead(
                        num_features=100,
                        num_actions=1,
                        head_architecture_cls=Linear,
                        head_architecture_cls_args={},
                        w_init=get_initializer(('zeros_', {})),
                        b_init=get_initializer(('zeros_', {})))
            }),
        rollout_len=ROLLOUT_LEN,
        extra_steps=EXTRA_STEPS)


def test_rollout_mgr_generate():
    rmgr = make_rollout_mgr()
    rollout_report = rmgr.generate()
    metadata = rollout_report.pop('metadata')
    assert metadata == {
        'ep_len': [],
        'ep_ret': [],
        'ep_len_raw': [],
        'ep_ret_raw': [],
    }
    tc.testing.assert_close(
        actual=rollout_report['observations'],
        expected=tc.arange(ROLLOUT_LEN + EXTRA_STEPS + 1))

    rollout_report2 = rmgr.generate()
    tc.testing.assert_close(
        actual=rollout_report2['observations'],
        expected=tc.tensor([7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
