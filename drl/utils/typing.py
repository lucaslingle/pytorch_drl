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

ActionType = Union[int, np.ndarray]
ObservationType = np.ndarray
RewardType = Union[float, Mapping[str, float]]
EnvStepOutput = Tuple[ObservationType, RewardType, bool, Mapping[str, Any]]