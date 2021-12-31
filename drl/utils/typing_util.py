from typing import Union

import torch as tc
import gym


Env = gym.core.Env
Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scheduler = tc.optim.lr_scheduler._LRScheduler
Checkpointable = Union[Module, Optimizer, Scheduler]
