from typing import Dict, Any, Union
import copy

import gym
import numpy as np
import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.envs.wrappers.stateless.abstract import Wrapper, RewardSpec
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.agents.preprocessing import ToChannelMajor
from drl.algos.common import global_mean
from drl.utils.optimization import get_optimizer


class RNDNetwork(tc.nn.Module):
    def __init__(self, data_shape, widening):
        super().__init__()
        self._input_channels = data_shape[-1]
        self._widening = widening
        self._network = tc.nn.Sequential(
            ToChannelMajor(),
            tc.nn.Conv2d(self._input_channels, 16 * widening, (8,8), (4,4), (0,0)),
            tc.nn.LeakyReLU(negative_slope=0.2),
            tc.nn.Conv2d(16 * widening, 32 * widening, (4,4), (2,2), (0,0)),
            tc.nn.LeakyReLU(negative_slope=0.2),
            tc.nn.Conv2d(32 * widening, 32 * widening, (4,4), (1,1), (0,0)),
            tc.nn.LeakyReLU(negative_slope=0.2),
            tc.nn.Flatten()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._network:
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self._network(x)


class TeacherNetwork(tc.nn.Module):
    def __init__(self, data_shape):
        super().__init__()
        self._data_shape = data_shape
        self._network = tc.nn.Sequential(
            RNDNetwork(data_shape=data_shape, widening=1),
            tc.nn.Linear(1152, 512)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._network:
            if isinstance(m, tc.nn.Linear):
                tc.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self._network(x)


class StudentNetwork(tc.nn.Module):
    def __init__(self, data_shape, widening=1):
        super().__init__()
        self._network = tc.nn.Sequential(
            RNDNetwork(data_shape, widening=widening),
            tc.nn.Linear(1152 * widening, 256 * widening),
            tc.nn.ReLU(),
            tc.nn.Linear(256 * widening, 256 * widening),
            tc.nn.ReLU(),
            tc.nn.Linear(256 * widening, 512)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._network:
            if isinstance(m, tc.nn.Linear):
                tc.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self._network(x)


class RandomNetworkDistillationWrapper(TrainableWrapper):
    def __init__(
            self,
            env: Union[gym.core.Env, Wrapper],
            rnd_optimizer_cls_name: str,
            rnd_optimizer_args: Dict[str, Any],
            world_size: int,
            widening: int,
            non_learning_steps: int
    ):
        super().__init__(env)
        self._data_shape = (84, 84, 1)
        self._world_size = world_size
        self._synced_normalizer = Normalizer(self._data_shape, -5, 5)
        self._unsynced_normalizer = Normalizer(self._data_shape, -5, 5)
        self._teacher_net = DDP(TeacherNetwork(self._data_shape))
        self._student_net = DDP(StudentNetwork(self._data_shape, widening))
        self._optimizer = get_optimizer(
            model=self._student_net,
            cls_name=rnd_optimizer_cls_name,
            cls_args=rnd_optimizer_args)
        self._non_learning_steps = non_learning_steps
        self._reward_name = 'intrinsic_rnd'
        self._reward_spec = self._get_reward_spec()
        self._run_checks()

    @property
    def reward_name(self):
        return self._reward_name

    def _run_checks(self):
        space = self.env.observation_space
        cond1 = str(space.dtype) == 'float32'
        cond2 = space.shape[0:2] == self._data_shape[0:2]
        if not cond1:
            msg = "Attempted to wrap env with non-float32 obs dtype."
            raise TypeError(msg)
        if not cond2:
            msg = f"Attempted to wrap env with unsupported shape {space.shape}."
            raise ValueError(msg)

    def _get_reward_spec(self):
        parent_reward_spec = self.env.reward_spec
        if parent_reward_spec is None:
            reward_keys = ['extrinsic_raw', 'extrinsic', self._reward_name]
        else:
            reward_keys = copy.deepcopy(parent_reward_spec.keys)
            reward_keys.append(self._reward_name)
        return RewardSpec(keys=reward_keys)

    def _sync_normalizers_global(self):
        self._synced_normalizer.steps = global_mean(
            self._unsynced_normalizer.steps, self._world_size)
        self._synced_normalizer.moment1 = global_mean(
            self._unsynced_normalizer.moment1, self._world_size)
        self._synced_normalizer.moment2 = global_mean(
            self._unsynced_normalizer.moment2, self._world_size)

    def _sync_normalizers_local(self):
        self._unsynced_normalizer.steps = self._synced_normalizer.steps
        self._unsynced_normalizer.moment1 = self._synced_normalizer.moment1
        self._unsynced_normalizer.moment2 = self._synced_normalizer.moment2

    def get_checkpointables(self):
        checkpointables = dict()
        if isinstance(self.env, Wrapper):
            checkpointables.update(self.env.get_checkpointables())
        checkpointables.update({
            'rnd_observation_normalizer': self._synced_normalizer,
            'rnd_teacher_net': self._teacher_net,
            'rnd_student_net': self._student_net,
            'rnd_optimizer': self._optimizer
        })
        return checkpointables

    def step(self, ac):
        if self._unsynced_normalizer.steps < self._synced_normalizer.steps:
            self._sync_normalizers_local()
        obs, rew, done, info = self.env.step(ac)
        obs = tc.tensor(obs).float()[:, :, -1:]
        normalized = self._synced_normalizer(obs.unsqueeze(0))
        _ = self._unsynced_normalizer.update(obs)
        y, yhat = self._teacher_net(normalized), self._student_net(normalized)
        rewards_dict = {self._reward_name: tc.square(y-yhat).sum(dim=-1).item()}
        if isinstance(rew, dict):
            rewards_dict.update(rew)
        else:
            rewards_dict.update({'extrinsic_raw': rew, 'extrinsic': rew})
        return obs, rewards_dict, done, info

    def learn(self, obs_batch, **kwargs):
        if self._synced_normalizer.steps >= self._non_learning_steps:
            self._optimizer.zero_grad()
            normalized = self._synced_normalizer(obs_batch)
            y, yhat = self._teacher_net(normalized), self._student_net(normalized)
            loss = tc.square(y-yhat).sum(dim=-1).mean(dim=0)
            loss.backward()
            self._optimizer.step()
        self._sync_normalizers_global()
        self._sync_normalizers_local()
