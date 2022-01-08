from collections import Counter, deque

import torch as tc
import numpy as np


def global_mean(metric, world_size, item=False):
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    if item and np.prod(global_metric.shape) == 1:
        global_metric = global_metric.item()
    return global_metric / world_size


def global_means(metrics, world_size, item):
    return Counter({
        k: global_mean(v, world_size, item) for k,v in metrics.items()
    })


def global_gather(field_values, world_size):
    output = [None for _ in range(world_size)]
    tc.distributed.all_gather_object(output, field_values)
    output = [element for lst in output for element in lst]
    return output


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
