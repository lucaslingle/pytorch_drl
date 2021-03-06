from typing import Mapping, Any
from collections import deque

import torch as tc

from drl.algos.common.metrics import (
    global_mean, global_means, global_gather, global_gathers, MultiQueue)
from drl.utils.test_distributed import requires_process_group, WORLD_SIZE


@requires_process_group
def local_test_global_mean(rank: int, **kwargs: Mapping[str, Any]) -> None:
    local_value = rank * tc.tensor([1., 2., 3.])
    global_value_actual = global_mean(
        local_value=local_value, world_size=WORLD_SIZE, item=False)
    global_value_expected = 0.5 * tc.tensor([1., 2., 3.])
    if rank == 0:
        tc.testing.assert_close(
            actual=global_value_actual, expected=global_value_expected)


def test_global_mean() -> None:
    port = 12001
    tc.multiprocessing.spawn(
        local_test_global_mean, args=(port, ), nprocs=WORLD_SIZE, join=True)


@requires_process_group
def local_test_global_means(rank: int, **kwargs: Mapping[str, Any]) -> None:
    local_values = {
        'foo': rank * tc.tensor([1., 2., 3.]),
        'bar': -2 * rank * tc.tensor([1., 2., 3.])
    }
    global_values_actual = global_means(
        local_values=local_values, world_size=WORLD_SIZE, item=False)
    global_values_expected = {
        'foo': 0.5 * tc.tensor([1., 2., 3.]),
        'bar': -1.0 * tc.tensor([1., 2., 3.])
    }
    if rank == 0:
        for name in local_values.keys():
            tc.testing.assert_close(
                actual=global_values_actual[name],
                expected=global_values_expected[name])


def test_global_means() -> None:
    port = 12002
    tc.multiprocessing.spawn(
        local_test_global_means, args=(port, ), nprocs=WORLD_SIZE, join=True)


@requires_process_group
def local_test_global_gather(rank: int, **kwargs: Mapping[str, Any]) -> None:
    local_list = [rank * tc.tensor([1., 2., 3.])]
    global_list_actual = global_gather(
        local_list=local_list, world_size=WORLD_SIZE)
    if rank == 0:
        global_list_expected_upto_permute = [
            j * tc.tensor([1., 2., 3.]) for j in range(WORLD_SIZE)
        ]
        if global_list_actual[0].sum().item() == 0.0:
            global_list_expected = global_list_expected_upto_permute
        else:
            global_list_expected = global_list_expected_upto_permute[::-1]
        for j in range(WORLD_SIZE):
            tc.testing.assert_close(
                actual=global_list_actual[j], expected=global_list_expected[j])


def test_global_gather() -> None:
    port = 12003
    tc.multiprocessing.spawn(
        local_test_global_gather, args=(port, ), nprocs=WORLD_SIZE, join=True)


@requires_process_group
def local_test_global_gathers(rank: int, **kwargs: Mapping[str, Any]) -> None:
    local_lists = {
        'foo': [rank * tc.tensor([1., 2., 3.])],
        'bar': [-2 * rank * tc.tensor([1., 2., 3.])]
    }
    global_lists_actual = global_gathers(
        local_lists=local_lists, world_size=WORLD_SIZE)

    if rank == 0:
        global_lists_expected_upto_permute = {
            'foo': [j * tc.tensor([1., 2., 3.]) for j in range(WORLD_SIZE)],
            'bar': [
                -2 * j * tc.tensor([1., 2., 3.]) for j in range(WORLD_SIZE)
            ]
        }
        for name in local_lists.keys():
            global_list_actual = global_lists_actual[name]
            global_list_expected_upto = global_lists_expected_upto_permute[name]
            if global_list_actual[0].sum().item() == 0.0:
                global_list_expected = global_list_expected_upto
            else:
                global_list_expected = global_list_expected_upto[::-1]
            for j in range(WORLD_SIZE):
                tc.testing.assert_close(
                    actual=global_list_actual[j],
                    expected=global_list_expected[j])


def test_global_gathers() -> None:
    port = 12004
    tc.multiprocessing.spawn(
        local_test_global_gathers, args=(port, ), nprocs=WORLD_SIZE, join=True)


def test_multiqueue_empty():
    mq = MultiQueue(maxlen=2)
    assert list(mq.keys()) == []


def test_multiqueue_get():
    mq = MultiQueue(maxlen=2)
    q = deque(maxlen=2)
    q.extend([1, 2])
    mq._queues['foo'] = q
    assert mq.get('foo') == deque([1, 2], maxlen=2)


def test_multiqueue_keys():
    mq = MultiQueue(maxlen=2)
    q = deque(maxlen=2)
    q.extend([1, 2])
    mq._queues['foo'] = q
    assert set(mq.keys()) == {'foo'}


def test_multiqueue_items():
    mq = MultiQueue(maxlen=2)
    q = deque(maxlen=2)
    q.extend([1, 2])
    mq._queues['foo'] = q
    assert list(mq.items(mean=False))[0] == ('foo', deque([1, 2], maxlen=2))
    assert list(mq.items(mean=True))[0] == ('foo', 1.5)


def test_multiqueue_update():
    mq = MultiQueue(maxlen=2)
    mq.update({'foo': [1, 2], 'bar': [0.0, -1.0]})
    assert set(mq.keys()) == {'foo', 'bar'}
    assert mq.get('foo') == deque([1, 2], maxlen=2)
    assert mq.get('bar') == deque([0.0, -1.0], maxlen=2)
    mq.update({'foo': [3], 'bar': [1.0], 'baz': [3.0]})
    assert set(mq.keys()) == {'foo', 'bar', 'baz'}
    assert mq.get('foo') == deque([2, 3], maxlen=2)
    assert mq.get('bar') == deque([-1.0, 1.0], maxlen=2)
    assert mq.get('baz') == deque([3.0], maxlen=2)


def test_multiqueue_mean():
    mq = MultiQueue(maxlen=2)
    mq.update({'foo': [1, 2], 'bar': [0.0, -1.0]})
    assert set(mq.keys()) == {'foo', 'bar'}
    assert mq.mean('foo') == 1.5
    assert mq.mean('bar') == -0.5
    mq.update({'foo': [3], 'bar': [1.0], 'baz': [3.0]})
    assert mq.mean('foo') == 2.5
    assert mq.mean('bar') == 0.0
    assert mq.mean('baz') == 3.0
