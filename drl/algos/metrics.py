from collections import Counter

import torch as tc


def global_mean(metric, world_size):
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.item() / world_size


def global_means(metrics, world_size):
    # for logging purposes only!
    return Counter({k: global_mean(v, world_size) for k, v in metrics.items()})
