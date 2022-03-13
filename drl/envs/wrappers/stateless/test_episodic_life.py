from drl.envs.testing import LockstepEnv
from drl.envs.wrappers.stateless.episodic_life import EpisodicLifeWrapper


def test_episodic_life():
    env = LockstepEnv(cardinality=3)

    wrapped = EpisodicLifeWrapper(
        env, lives_fn=lambda env: int(env._state != 0) + 1, noop_action=42)
    # noop action only matters at reset,
    # so as long as the state doesnt step we're good.
    _ = wrapped.reset()
    o_tp1, r_t, d_t, i_t = wrapped.step(0)
    assert not d_t
    o_tp1, r_t, d_t, i_t = wrapped.step(1)
    assert not d_t
    o_tp1, r_t, d_t, i_t = wrapped.step(2)
    assert d_t
