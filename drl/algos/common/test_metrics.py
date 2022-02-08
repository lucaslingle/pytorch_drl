import os
from collections import deque

import pytest
import torch as tc
from torch.multiprocessing.spawn import ProcessRaisedException

from drl.algos.common.metrics import (
    global_mean, global_means, global_gather, global_gathers, MultiQueue)

WORLD_SIZE = 2


def make_process_group(rank: int, port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    tc.distributed.init_process_group(
        backend='gloo', world_size=WORLD_SIZE, rank=rank)


def destroy_process_group() -> None:
    return tc.distributed.destroy_process_group()


def local_vanilla_false_assert(rank, port):
    make_process_group(rank, port)
    if rank == 0:
        assert 2 + 2 == 5
    destroy_process_group()


def test_vanilla_false_assert() -> None:
    with pytest.raises(ProcessRaisedException):
        tc.multiprocessing.spawn(
            local_vanilla_false_assert,
            args=(11999, ),
            nprocs=WORLD_SIZE,
            join=True)


def local_torch_false_assert(rank: int, port: int) -> None:
    make_process_group(rank, port)
    if rank == 0:
        tc.testing.assert_close(
            actual=tc.tensor([1.]), expected=tc.tensor([0.]))
    destroy_process_group()


def test_torch_false_assert() -> None:
    with pytest.raises(ProcessRaisedException):
        tc.multiprocessing.spawn(
            local_torch_false_assert,
            args=(12000, ),
            nprocs=WORLD_SIZE,
            join=True)


def local_test_global_mean(rank: int, port: int) -> None:
    make_process_group(rank, port)
    local_value = rank * tc.tensor([1., 2., 3.])
    global_value_actual = global_mean(
        local_value=local_value, world_size=WORLD_SIZE, item=False)
    global_value_expected = 0.5 * tc.tensor([1., 2., 3.])
    if rank == 0:
        tc.testing.assert_close(
            actual=global_value_actual, expected=global_value_expected)
    destroy_process_group()


def test_global_mean() -> None:
    tc.multiprocessing.spawn(
        local_test_global_mean, args=(12001, ), nprocs=WORLD_SIZE, join=True)


def local_test_global_means(rank: int, port: int) -> None:
    make_process_group(rank, port)
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
    destroy_process_group()


def test_global_means() -> None:
    tc.multiprocessing.spawn(
        local_test_global_means, args=(12002, ), nprocs=WORLD_SIZE, join=True)


def local_test_global_gather(rank: int, port: int) -> None:
    make_process_group(rank, port)
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
    destroy_process_group()


def test_global_gather() -> None:
    tc.multiprocessing.spawn(
        local_test_global_gather, args=(12003, ), nprocs=WORLD_SIZE, join=True)


def local_test_global_gathers(rank: int, port: int) -> None:
    make_process_group(rank, port)
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
    destroy_process_group()


def test_global_gathers() -> None:
    tc.multiprocessing.spawn(
        local_test_global_gathers, args=(12004, ), nprocs=WORLD_SIZE, join=True)


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


def test_multiqueue_mean():
    mq = MultiQueue(maxlen=2)
    mq.update({'foo': [1, 2], 'bar': [0.0, -1.0]})
    assert set(mq.keys()) == {'foo', 'bar'}
    assert mq.mean('foo') == 1.5
    assert mq.mean('bar') == -0.5
