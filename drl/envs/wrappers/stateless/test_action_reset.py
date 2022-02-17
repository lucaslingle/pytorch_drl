from drl.envs.testing import LockstepEnv
from drl.envs.wrappers.stateless.action_reset import ActionResetWrapper


def test_action_reset():
    env = LockstepEnv()
    obs_actual = env.reset()
    assert obs_actual == 0

    wrapped = ActionResetWrapper(env, action_sequence=[0, 1, 2, 3])
    obs_actual = wrapped.reset()
    assert obs_actual == 4
