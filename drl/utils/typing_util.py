from typing import Union

import torch as tc
import gym

from drl.envs.wrappers.common.abstract import Wrapper


Env = Union[gym.core.Env, Wrapper]
Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scheduler = tc.optim.lr_scheduler._LRScheduler
Checkpointable = Union[Module, Optimizer, Scheduler]
