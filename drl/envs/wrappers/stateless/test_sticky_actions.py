import gym

from drl.envs.wrappers.stateless.sticky_actions import StickyActionsWrapper

STICK_PROB = 0.25


def make_wrapped():
    env = gym.make('BreakoutNoFrameskip-v4')
    wrapped = StickyActionsWrapper(env, stick_prob=STICK_PROB, noop_action=0)
    return wrapped


def test_sticky_action_last_action_getter_setter():
    wrapped = make_wrapped()
    wrapped.last_action = 1
    assert wrapped.last_action == 1
    wrapped.last_action = 2
    assert wrapped.last_action == 2


def test_sticky_action_logic():
    wrapped = make_wrapped()
    _ = wrapped.reset()

    wrapped.last_action = 0
    actual = wrapped.logic(action=1, u=STICK_PROB * 0.5)
    assert actual == 0
    actual = wrapped.logic(action=1, u=STICK_PROB * 2.0)
    assert actual == 1

    wrapped.last_action = 1
    actual = wrapped.logic(action=2, u=STICK_PROB * 0.5)
    assert actual == 1
    actual = wrapped.logic(action=2, u=STICK_PROB * 2.0)
    assert actual == 2


def test_sticky_action_action():
    wrapped = make_wrapped()
    _ = wrapped.reset()
    num_stuck = 0
    num_total = 100000
    for _ in range(num_total):
        last_action = wrapped.last_action
        input_action = (last_action + 1) % wrapped.action_space.n
        maybe_sticky_action = wrapped.action(action=input_action)
        if maybe_sticky_action == last_action:
            num_stuck += 1
    frac_stuck = num_stuck / num_total
    delta95_for_n1000 = 1.96 * (frac_stuck * (1 - frac_stuck) / 100)**0.5
    assert frac_stuck - delta95_for_n1000 <= STICK_PROB
    assert STICK_PROB <= frac_stuck + delta95_for_n1000
