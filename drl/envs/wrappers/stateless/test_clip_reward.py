from drl.envs.testing import LockstepEnv
from drl.envs.wrappers.stateless.clip_reward import ClipRewardWrapper


def test_clip_reward():
    env = LockstepEnv()

    wrapped = ClipRewardWrapper(env, low=0.0, high=0.5, key='extrinsic')
    _ = wrapped.reset()
    o_tp1, r_t, d_t, i_t = wrapped.step(0)
    assert r_t['extrinsic'] == 0.5

    wrapped2 = ClipRewardWrapper(wrapped, low=0.0, high=0.25, key='extrinsic')
    _ = wrapped2.reset()
    o_tp1, r_t, d_t, i_t = wrapped2.step(0)
    assert r_t['extrinsic'] == 0.25
