import os

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
