import os
import functools

import pytest
import torch as tc
from torch.multiprocessing.spawn import ProcessRaisedException

WORLD_SIZE = 2


def make_process_group(rank: int, port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    tc.distributed.init_process_group(
        backend='gloo', world_size=WORLD_SIZE, rank=rank)


def destroy_process_group() -> None:
    return tc.distributed.destroy_process_group()


def requires_process_group(func):
    @functools.wraps(func)
    def wrapped(rank, port, **kwargs):
        make_process_group(rank, port)
        kwargs.update({'rank': rank, 'port': port})
        func(**kwargs)
        destroy_process_group()

    return wrapped


@requires_process_group
def local_vanilla_false_assert():
    assert 2 + 2 == 5


def test_vanilla_false_assert():
    with pytest.raises(ProcessRaisedException):
        tc.multiprocessing.spawn(
            local_vanilla_false_assert,
            args=(11999, ),
            nprocs=WORLD_SIZE,
            join=True)


@requires_process_group
def local_torch_false_assert():
    tc.testing.assert_close(actual=tc.tensor([1.]), expected=tc.tensor([0.]))


def test_torch_false_assert():
    with pytest.raises(ProcessRaisedException):
        tc.multiprocessing.spawn(
            local_torch_false_assert,
            args=(12000, ),
            nprocs=WORLD_SIZE,
            join=True)
