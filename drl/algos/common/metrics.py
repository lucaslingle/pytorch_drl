from typing import Union, Mapping, List, Any
from collections import Counter, deque

import torch as tc
import numpy as np


def global_mean(local_value: tc.Tensor,
                world_size: int,
                item: bool = False) -> Union[tc.Tensor, float]:
    """
    Takes the global mean of a tensor across processes.

    Args:
        local_value (torch.Tensor): Torch tensor containing a metric.
        world_size (int): Number of processes.
        item (bool): Whether to call the 'item' method on the global-mean tensor.

    Returns:
        (Union[torch.Tensor, float]): Torch tensor or float.
    """
    global_value = local_value.clone().float().detach()
    tc.distributed.all_reduce(global_value, op=tc.distributed.ReduceOp.SUM)
    if item and np.prod(global_value.shape) == 1:
        global_value = global_value.item()
    return global_value / world_size


def global_means(
        local_values: Mapping[str, tc.Tensor], world_size: int,
        item: bool) -> Counter:
    """
    Performs a global_mean on each tensor in a mapping of names to Torch tensors.

    Args:
        local_values (Mapping[str, tc.Tensor]): Dict of Torch tensors,
             keyed by name.
        world_size (int): Number of processes.
        item (bool): Whether to call the 'item' method on the global-mean tensor.

    Returns:
        collections.Counter: Counter instance storing Torch tensors or floats.
    """
    return Counter(
        {k: global_mean(v, world_size, item) for k, v in local_values.items()})


def global_gather(local_list: List[Any], world_size: int) -> List[Any]:
    """
    Gathers all items into a list from across processes.

    Args:
        local_list (List[Any]): List of items.
        world_size (int): Number of processes.

    Returns:
        List[Any]: List of values gathered.
    """
    output = [None for _ in range(world_size)]
    tc.distributed.all_gather_object(output, local_list)
    output = [element for lst in output for element in lst]
    return output


def global_gathers(local_lists: Mapping[str, List[Any]],
                   world_size: int) -> Mapping[str, List[Any]]:
    """
    Performs a global_gather on each item in a mapping from names to lists.

    Args:
        local_lists (Mapping[str, List[Any]]): Dict of list of items,
            keyed by name.
        world_size (int): Number of processes.

    Returns:
        collections.Counter: Counter storing lists of Torch tensors or floats.
    """
    return Counter(
        {k: global_gather(v, world_size) for k, v in local_lists.items()})


def pretty_print(metrics: Union[Mapping[str, Any], 'MultiQueue']) -> None:
    """
    Pretty prints the provided metrics to stdout.

    Args:
        metrics (Mapping[str, Any]): Metrics.

    Returns:
        None.
    """
    print("-" * 100)
    maxlen_name_len = max(len(name) for name in metrics.keys())
    for name, value in metrics.items():
        blankspace = " " * (maxlen_name_len - len(name) + 1)
        print(f"{name}: {blankspace}{value:>0.6f}")
    print("-" * 100)


class MultiQueue:
    """
    A data-structure containing queues to track running stats,
    which possibly arrive at different or variable rates.
    """
    def __init__(self, memory_len: int):
        """
        Args:
            memory_len (int): Capacity of each deque.
        """
        self._memory_len = memory_len
        self._queues = dict()

    def _update_field(self, field, values):
        if field not in self._queues:
            self._queues[field] = deque(maxlen=self._memory_len)
        self._queues[field].extend(values)

    def update(self, new_stats: Mapping[str, List[Union[int, float]]]) -> None:
        """
        Updates the queues with the provided new values.

        Args:
            new_stats (Mapping[str, List[Union[int, float]]]): A dictionary of new values,
                keyed by field name, each mapping to a list of new stats for that key.

        Returns:
            None:
        """
        for field in new_stats:
            self._update_field(field, new_stats[field])

    def mean(self, field: str) -> float:
        """
        Gets the mean value in the queue for the field provided.

        Args:
            field (str): Field to compute mean for.

        Returns:
            float: mean value in the field's queue.
        """
        if len(self._queues[field]) == 0:
            return np.nan
        return sum(self._queues[field]) / len(self._queues[field])

    def keys(self):
        return iter(field for field in self._queues)

    def items(self, mean=True):
        """
        Gets the queues as an iterator, or their means.

        Args:
            mean (bool): Return the means? Default: True.

        Returns:
            Iterable[Tuple[str, Union[float, collections.deque]]]: Iterator of tuples,
                containing field names and either queues or their means.

        """
        return iter((field, self.mean(field) if mean else self._queues[field])
                    for field in self._queues)
