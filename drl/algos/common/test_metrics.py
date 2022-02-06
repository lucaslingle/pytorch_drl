import os

import torch as tc

from drl.algos.common.metrics import (
    global_mean, global_means, global_gather, global_gathers)


WORLD_SIZE = 2


def make_process_group(rank: int, port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    tc.distributed.init_process_group(
        backend='gloo',
        world_size=WORLD_SIZE,
        rank=rank)


def destroy_process_group() -> None:
    return tc.distributed.destroy_process_group()


def local_test_global_mean(rank, port):
    make_process_group(rank, port)
    local_value = rank * tc.tensor([1., 2., 3.])
    global_value_actual = global_mean(
        local_value=local_value, world_size=WORLD_SIZE, item=False)
    global_value_expected = 0.5 * tc.tensor([1., 2., 3.])
    if rank == 0:
        tc.testing.assert_close(
            actual=global_value_actual, expected=global_value_expected)
    destroy_process_group()


def test_global_mean():
    tc.multiprocessing.spawn(
        local_test_global_mean,
        args=(12001,),
        nprocs=WORLD_SIZE,
        join=True)


def local_test_global_means(rank, port):
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


def test_global_means():
    tc.multiprocessing.spawn(
        local_test_global_means,
        args=(12002,),
        nprocs=WORLD_SIZE,
        join=True)


def local_test_global_gather(rank, port):
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


def test_global_gather():
    tc.multiprocessing.spawn(
        local_test_global_gather,
        args=(12003,),
        nprocs=WORLD_SIZE,
        join=True)


def local_test_global_gathers(rank, port):
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
            'bar': [-2 * j * tc.tensor([1., 2., 3.]) for j in range(WORLD_SIZE)]
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


def test_global_gathers():
    tc.multiprocessing.spawn(
        local_test_global_gathers,
        args=(12004,),
        nprocs=WORLD_SIZE,
        join=True)
