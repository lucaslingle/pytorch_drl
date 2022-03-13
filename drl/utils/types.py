"""
Typing util.
"""

from typing import Union, Mapping, Any, Tuple

import torch as tc
import numpy as np

Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scheduler = tc.optim.lr_scheduler._LRScheduler
Checkpointable = Union[Module, Optimizer, Scheduler]

CreditAssignmentSpec = Mapping[str, Mapping[str, Union[str, Mapping[str, Any]]]]

Action = Union[int, np.ndarray]
Observation = np.ndarray
ScalarReward = float
DictReward = Mapping[str, float]
Reward = Union[ScalarReward, DictReward]
EnvOutput = Tuple[Observation, Reward, bool, Mapping[str, Any]]

FlatGrad = np.ndarray

NestedTensor = Union[tc.Tensor, 'NestedTensor']
Indices = Union[np.ndarray, slice]
