from collections import Counter, deque

import torch as tc
import numpy as np


def global_mean(metric, world_size):
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.item() / world_size


def global_means(metrics, world_size):
    return Counter({k: global_mean(v, world_size) for k,v in metrics.items()})


def global_gather(field_values, world_size):
    metadata_tensor = tc.tensor(field_values)
    shape, dtype = tuple(metadata_tensor.shape), metadata_tensor.dtype
    tensors_list = [tc.zeros(size=shape, dtype=dtype) for _ in range(world_size)]
    tc.distributed.all_gather(tensors_list, metadata_tensor)
    return list(tc.cat(tensors_list, dim=0).detach().numpy())


def global_gathers(metadata, world_size):
    return Counter({k: global_gather(v, world_size) for k,v in metadata.items()})


def pretty_print(metrics):
    print("-" * 100)
    maxlen_name_len = max(len(name) for name in metrics)
    for name, value in metrics.items():
        blankspace = " " * (maxlen_name_len - len(name) + 1)
        print(f"{name}: {blankspace}{value:>0.6f}")
    print("-" * 100)


class MultiDeque:
    def __init__(self, memory_len):
        self._memory_len = memory_len
        self._dequeues = dict()

    def __iter__(self):
        return iter([field for field in self._dequeues])

    def update_field(self, field, values):
        if field not in self._dequeues:
            self._dequeues[field] = deque(maxlen=self._memory_len)
        self._dequeues[field].extend(values)

    def update(self, new_metadata):
        for field in new_metadata:
            self.update_field(field, new_metadata[field])

    def mean(self, field):
        if len(self._dequeues[field]) == 0:
            return 0.0
        return np.mean(self._dequeues[field])

    def items(self):
        return iter([(field, self.mean(field)) for field in self._dequeues])
