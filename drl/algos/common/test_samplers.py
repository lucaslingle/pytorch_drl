import pytest
import torch as tc
import numpy as np

from drl.algos.common import IndicesIterator, IndependentSampler, TBPTTSampler
from drl.utils.nested import slice_nested_tensor


def test_indices_iterator() -> None:
    indices, stepsize = np.arange(9), 3

    indices_iterator = IndicesIterator(indices=indices, stepsize=stepsize)
    indices_so_far = set()
    for some_indices in indices_iterator:
        # test subset disjointness
        assert len(indices_so_far.intersection(set(some_indices))) == 0
        indices_so_far |= set(some_indices)
    # test set coverage
    assert indices_so_far == set(indices)

    indices_iterator2 = IndicesIterator(indices=indices, stepsize=stepsize)
    iter = indices_iterator2.__iter__()
    _ = iter.__next__()
    _ = iter.__next__()
    _ = iter.__next__()
    with pytest.raises(StopIteration):
        _ = iter.__next__()


def test_independent_sampler_on_list() -> None:
    list_ = [1, 2, 3, 4, 5, 6]

    sampler = IndependentSampler(rollout_len=6, batch_size=6)
    for indices in sampler:
        assert set([list_[idx] for idx in indices]) == set(list_)

    sampler2 = IndependentSampler(rollout_len=6, batch_size=2)
    seen_so_far = set()  # list items seen so far
    for indices in sampler2:
        sampled = set([list_[idx] for idx in indices])  # no duplicates anyway
        # test disjointness
        assert len(seen_so_far.intersection(sampled)) == 0
        seen_so_far |= sampled
    # test set coverage
    assert seen_so_far == set(list_)


def test_tbptt_sampler_on_list() -> None:
    list_ = [1, 2, 3, 4, 5, 6]

    sampler = TBPTTSampler(rollout_len=6, tbptt_len=6)
    for indices in sampler:
        assert set([list_[idx] for idx in indices]) == set(list_)

    sampler2 = TBPTTSampler(rollout_len=6, tbptt_len=2)
    seen_so_far = set()  # list items seen so far
    for indices in sampler2:
        sublist = [list_[idx] for idx in indices]
        # test per-truncation segment order preservation
        assert sublist == sorted(sublist)
        # test disjointness
        sampled = set(sublist)  # no duplicates anyway
        assert len(seen_so_far.intersection(sampled)) == 0
        seen_so_far |= sampled
    # test set coverage
    assert seen_so_far == set(list_)


def test_independent_sampler_on_torch_tensor() -> None:
    tensor_ = tc.tensor([1, 2, 3, 4, 5, 6])

    sampler = IndependentSampler(rollout_len=6, batch_size=6)
    for indices in sampler:
        assert set(tensor_[indices].detach().numpy()) == \
               set(tensor_.detach().numpy())

    sampler2 = IndependentSampler(rollout_len=6, batch_size=2)
    seen_so_far = set()  # list items seen so far
    for indices in sampler2:
        sampled = set(tensor_[indices].detach().numpy())  # no duplicates anyway
        # test disjointness
        assert len(seen_so_far.intersection(sampled)) == 0
        seen_so_far |= sampled
    # test set coverage
    assert seen_so_far == set(tensor_.detach().numpy())


def test_tbptt_sampler_on_torch_tensor() -> None:
    tensor_ = tc.tensor([1, 2, 3, 4, 5, 6])

    sampler = TBPTTSampler(rollout_len=6, tbptt_len=6)
    for indices in sampler:
        assert set(tensor_[indices].detach().numpy()) == \
               set(tensor_.detach().numpy())

    sampler2 = TBPTTSampler(rollout_len=6, tbptt_len=2)
    seen_so_far = set()  # list items seen so far
    for indices in sampler2:
        subtensor = tensor_[indices]
        # test per-truncation segment order preservation
        actual = subtensor
        expected, _ = tc.sort(subtensor)
        tc.testing.assert_close(actual=actual, expected=expected)
        # test disjointness
        sampled = set(subtensor.detach().numpy())  # no duplicates anyway
        assert len(seen_so_far.intersection(sampled)) == 0
        seen_so_far |= sampled
    # test set coverage
    assert seen_so_far == set(tensor_.detach().numpy())


def test_independent_sampler_on_nested_torch_tensor() -> None:
    nested_tensor = {
        'actions': tc.arange(6), 'observations': 2 * tc.arange(6).float()
    }

    sampler = IndependentSampler(rollout_len=6, batch_size=6)
    for indices in sampler:
        nested_subtensor = slice_nested_tensor(nested_tensor, indices)
        for key in nested_tensor:
            assert set(nested_subtensor[key].detach().numpy()) == \
                   set(nested_tensor[key].detach().numpy())

    sampler2 = IndependentSampler(rollout_len=6, batch_size=2)
    seen_so_far = {key: set() for key in nested_tensor}  # items seen so far
    for indices in sampler2:
        nested_subtensor = slice_nested_tensor(nested_tensor, indices)
        # test disjointness
        sampled = {
            key: set(nested_subtensor[key].detach().numpy())
            for key in nested_tensor
        }
        for key in nested_tensor:
            if key in seen_so_far:
                assert len(seen_so_far[key].intersection(sampled[key])) == 0
            seen_so_far[key] |= sampled[key]
    # test set coverage
    for key in nested_tensor:
        assert seen_so_far[key] == set(nested_tensor[key].detach().numpy())


def test_tbptt_sampler_on_nested_torch_tensor() -> None:
    nested_tensor = {
        'actions': tc.arange(6), 'observations': 2 * tc.arange(6).float()
    }

    sampler = TBPTTSampler(rollout_len=6, tbptt_len=6)
    for indices in sampler:
        nested_subtensor = slice_nested_tensor(nested_tensor, indices)
        for key in nested_tensor:
            assert set(nested_subtensor[key].detach().numpy()) == \
                   set(nested_tensor[key].detach().numpy())

    sampler2 = TBPTTSampler(rollout_len=6, tbptt_len=2)
    seen_so_far = {key: set() for key in nested_tensor}  # items seen so far
    for indices in sampler2:
        # test per-truncation segment order preservation
        nested_subtensor = slice_nested_tensor(nested_tensor, indices)
        for key in nested_tensor:
            actual = nested_subtensor[key]
            expected, _ = tc.sort(nested_subtensor[key])
            tc.testing.assert_close(actual=actual, expected=expected)
        # test disjointness
        sampled = {
            key: set(nested_subtensor[key].detach().numpy())
            for key in nested_tensor
        }  # no dupes anyway
        for key in nested_tensor:
            if key in seen_so_far:
                assert len(seen_so_far[key].intersection(sampled[key])) == 0
            seen_so_far[key] |= sampled[key]
    # test set coverage
    for key in nested_tensor:
        assert seen_so_far[key] == set(nested_tensor[key].detach().numpy())
