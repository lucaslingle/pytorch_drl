import torch as tc
import numpy as np

from drl.utils.nested import clone_nested_tensor, slice_nested_tensor


def make_nested_tensor():
    return {
        'foo': tc.arange(10), 'bar': {
            'baz': tc.zeros(10, dtype=tc.float32)
        }
    }


def check_same(nt, other):
    if isinstance(nt, tc.Tensor) and isinstance(other, tc.Tensor):
        tc.testing.assert_close(actual=nt, expected=other)
    elif isinstance(nt, dict) and isinstance(other, dict):
        assert set(nt.keys()) == set(other.keys())
        for k in nt:
            check_same(nt=nt[k], other=other[k])
    else:
        assert False


def test_clone_nested_tensor():
    nt = make_nested_tensor()
    nt_clone = clone_nested_tensor(nt)
    check_same(nt, nt_clone)


def test_slice_nested_tensor():
    nt = make_nested_tensor()
    nt_sliced_actual = slice_nested_tensor(nt, slice(0, 3))
    nt_sliced_expected = {
        'foo': tc.arange(3), 'bar': {
            'baz': tc.zeros(3, dtype=tc.float32)
        }
    }
    check_same(nt_sliced_actual, nt_sliced_expected)

    nt = make_nested_tensor()
    nt_sliced_actual = slice_nested_tensor(nt, np.array([1, 2, 5]))
    nt_sliced_expected = {
        'foo': tc.tensor([1, 2, 5]),
        'bar': {
            'baz': tc.zeros(3, dtype=tc.float32)
        }
    }
    check_same(nt_sliced_actual, nt_sliced_expected)
