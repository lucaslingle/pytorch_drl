from typing import Union, Mapping, List, Any
from collections import Counter, deque

import torch as tc
import numpy as np


def global_mean(
        metric: tc.Tensor,
        world_size: int,
        item: bool = False
) -> Union[tc.Tensor, float]:
    """
    Takes the global mean of a tensor across processes.

    Args:
        metric: Torch tensor.
        world_size: Number of processes.
        item: Whether to call the 'item' method on the global-mean tensor.

    Returns:
        Torch tensor or float.
    """
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    if item and np.prod(global_metric.shape) == 1:
        global_metric = global_metric.item()
    return global_metric / world_size


def global_means(
        metrics: Mapping[str, tc.Tensor],
        world_size: int,
        item: bool
) -> Mapping[str, Union[tc.Tensor, float]]:
    """
    Performs a global_mean on each tensor in a mapping of names to Torch tensors.

    Args:
        metrics: Mapping of Torch tensors keyed by name.
        world_size: Number of processes.
        item: Whether to call the 'item' method on the global-mean tensor.

    Returns:
        collections.Counter object storing Torch tensors or floats.
    """
    return Counter({
        k: global_mean(v, world_size, item) for k,v in metrics.items()
    })


def global_gather(field_values: List[Any], world_size: int) -> List[Any]:
    """
    Gathers all items into a list from across processes.

    Args:
        field_values: List of values.
        world_size: Number of processes.

    Returns:
        List of values gathered.
    """
    output = [None for _ in range(world_size)]
    tc.distributed.all_gather_object(output, field_values)
    output = [element for lst in output for element in lst]
    return output


def global_gathers(
        metadata: Mapping[str, List[Any]], world_size: int
) -> Mapping[str, List[Any]]:
    """
    Performs a global_gather on each item in a mapping from names to lists.

    Args:
        metadata: Mapping from field names to list of items.
        world_size: Number of processes.

    Returns:
        collections.Counter object storing lists of Torch tensors or floats.
    """
    return Counter({k: global_gather(v, world_size) for k,v in metadata.items()})


def pretty_print(metrics: Mapping[str, Union[int, float]]) -> None:
    print("-" * 100)
    maxlen_name_len = max(len(name) for name in metrics)
    for name, value in metrics.items():
        blankspace = " " * (maxlen_name_len - len(name) + 1)
        print(f"{name}: {blankspace}{value:>0.6f}")
    print("-" * 100)


class MultiDeque:
    def __init__(self, memory_len):
        self._memory_len = memory_len
        self._deques = dict()

    def __iter__(self):
        return iter(field for field in self._deques)

    def update_field(self, field, values):
        if field not in self._deques:
            self._deques[field] = deque(maxlen=self._memory_len)
        self._deques[field].extend(values)

    def update(self, new_metadata):
        for field in new_metadata:
            self.update_field(field, new_metadata[field])

    def mean(self, field):
        if len(self._deques[field]) == 0:
            return 0.0
        return sum(self._deques[field]) / len(self._deques[field])

    def items(self):
        return [(field, self.mean(field)) for field in self._deques]
