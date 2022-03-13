import torch as tc

from drl.envs.wrappers.stateful.normalize import Normalizer


def test_normalizer_getters_setters() -> None:
    data_shape = [84, 84, 4]
    normalizer = Normalizer(data_shape=data_shape, clip_low=-5., clip_high=5.)

    tc.testing.assert_close(normalizer.steps, tc.tensor(0.))
    normalizer.steps = tc.tensor(1.)
    tc.testing.assert_close(normalizer.steps, tc.tensor(1.))
    normalizer.steps = tc.tensor(0.)
    tc.testing.assert_close(normalizer.steps, tc.tensor(0.))

    tc.testing.assert_close(
        actual=normalizer.moment1,
        expected=tc.zeros(size=data_shape, dtype=tc.float32))
    normalizer.moment1 = tc.ones(size=data_shape, dtype=tc.float32)
    tc.testing.assert_close(
        actual=normalizer.moment1,
        expected=tc.ones(size=data_shape, dtype=tc.float32))
    normalizer.moment1 = tc.zeros(size=data_shape, dtype=tc.float32)
    tc.testing.assert_close(
        actual=normalizer.moment1,
        expected=tc.zeros(size=data_shape, dtype=tc.float32))

    tc.testing.assert_close(
        actual=normalizer.moment2,
        expected=tc.zeros(size=data_shape, dtype=tc.float32))
    normalizer.moment2 = tc.ones(size=data_shape, dtype=tc.float32)
    tc.testing.assert_close(
        actual=normalizer.moment2,
        expected=tc.ones(size=data_shape, dtype=tc.float32))
    normalizer.moment2 = tc.zeros(size=data_shape, dtype=tc.float32)
    tc.testing.assert_close(
        actual=normalizer.moment2,
        expected=tc.zeros(size=data_shape, dtype=tc.float32))


def test_normalizer_forward() -> None:
    data_shape = [84, 84, 4]
    normalizer = Normalizer(data_shape=data_shape, clip_low=-5., clip_high=5.)

    # m1 = 0, m2 = 0, input = 0, normalized = 0
    normalizer.moment1 = tc.zeros(size=data_shape, dtype=tc.float32)
    normalizer.moment2 = tc.zeros(size=data_shape, dtype=tc.float32)
    actual = normalizer(tc.zeros(size=[1, *data_shape], dtype=tc.float32))
    expected = tc.zeros(size=[1, *data_shape], dtype=tc.float32)
    tc.testing.assert_close(actual=actual, expected=expected)

    # m1 = 1, m2 = 1, input = 1, normalized = 0
    normalizer.moment1 = tc.ones(size=data_shape, dtype=tc.float32)
    normalizer.moment2 = tc.ones(size=data_shape, dtype=tc.float32)
    actual = normalizer(tc.ones(size=[1, *data_shape], dtype=tc.float32))
    expected = tc.zeros(size=[1, *data_shape], dtype=tc.float32)
    tc.testing.assert_close(actual=actual, expected=expected)

    # m1 = 1, m2 = 5, input = 3, normalized = 0.5
    normalizer.moment1 = tc.ones(size=data_shape, dtype=tc.float32)
    normalizer.moment2 = 5 * tc.ones(size=data_shape, dtype=tc.float32)
    actual = normalizer(3 * tc.ones(size=[1, *data_shape], dtype=tc.float32))
    expected = tc.ones(size=[1, *data_shape], dtype=tc.float32)
    tc.testing.assert_close(actual=actual, expected=expected)

    # m1 = 2, m2 = 6, input = 26, normalized = 12, clipped = 5
    normalizer.moment1 = 2 * tc.ones(size=data_shape, dtype=tc.float32)
    normalizer.moment2 = 6 * tc.ones(size=data_shape, dtype=tc.float32)
    actual = normalizer(26 * tc.ones(size=[1, *data_shape], dtype=tc.float32))
    expected = 5 * tc.ones(size=[1, *data_shape], dtype=tc.float32)
    tc.testing.assert_close(actual=actual, expected=expected)

    # m1 = 2, m2 = 6, input = -22, normalized = -12, clipped = -5
    normalizer.moment1 = 2 * tc.ones(size=data_shape, dtype=tc.float32)
    normalizer.moment2 = 6 * tc.ones(size=data_shape, dtype=tc.float32)
    actual = normalizer(-22 * tc.ones(size=[1, *data_shape], dtype=tc.float32))
    expected = -5 * tc.ones(size=[1, *data_shape], dtype=tc.float32)
    tc.testing.assert_close(actual=actual, expected=expected)
