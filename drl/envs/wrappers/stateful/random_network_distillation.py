from typing import Dict, Any

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.utils.typing_util import Env
from drl.envs.wrappers.common.abstract import Wrapper
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.agents.preprocessing import ToChannelMajor
from drl.algos.metrics import global_mean
from drl.utils.optim_util import get_optimizer


class TeacherNetwork(tc.nn.Module):
    """
    Teacher network for Random Network Distillation.
    Assumes input image has dimension 84x84x4.
    """
    def __init__(self, data_shape, width_multiplier=1):
        super().__init__()
        self._data_shape = data_shape
        self._network = tc.nn.Sequential(
            ToChannelMajor(),
        )

    def forward(self, x):
        raise NotImplementedError


class StudentNetwork(tc.nn.Module):
    """
    Student network for Random Network Distillation.
    Assumes input image has dimension 84x84x4.
    """
    def __init__(self, data_shape, width_multiplier=1):
        super().__init__()
        self._data_shape = data_shape
        self._network = tc.nn.Sequential(
            ToChannelMajor(),
        )

    def forward(self, x):
        raise NotImplementedError


class RandomNetworkDistillationWrapper(TrainableWrapper):
    def __init__(
            self,
            env: Env,
            rnd_optimizer_cls_name: str,
            rnd_optimizer_args: Dict[str, Any],
            world_size: int,
            width_multipler: int = 1
    ):
        super().__init__(env)
        self._data_shape = (84, 84, 4)
        self._world_size = world_size
        self._synced_normalizer = Normalizer(self._data_shape, -5, 5)
        self._unsynced_normalizer = Normalizer(self._data_shape, -5, -5)
        self._teacher_net = DDP(TeacherNetwork(self._data_shape, width_multipler))
        self._student_net = DDP(StudentNetwork(self._data_shape, width_multipler))
        self._optimizer = get_optimizer(
            model=self._student_net,
            optimizer_cls_name=rnd_optimizer_cls_name,
            optimizer_args=rnd_optimizer_args)
        self._run_checks()
        self._sync_normalizers()

    def _run_checks(self):
        space = self.env.observation_space
        cond1 = str(space.dtype) == 'float32'
        cond2 = space.shape == self._data_shape
        if not cond1:
            msg = "Attempted to wrap env with non-float32 obs dtype."
            raise TypeError(msg)
        if not cond2:
            msg = f"Attempted to wrap env with unsupported shape {space.shape}."
            raise ValueError(msg)

    def _sync_normalizers(self):
        self._synced_normalizer.steps = global_mean(
            self._unsynced_normalizer.steps, self._world_size)
        self._synced_normalizer.mean = global_mean(
            self._unsynced_normalizer.mean, self._world_size)
        self._synced_normalizer.stddev = global_mean(
            self._unsynced_normalizer.stddev, self._world_size)
        self._unsynced_normalizer.steps = self._synced_normalizer.steps
        self._unsynced_normalizer.mean = self._synced_normalizer.mean
        self._unsynced_normalizer.stddev = self._synced_normalizer.stddev

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
        obs, rew, done, info = self.env.step(ac)
        normalized = self._synced_normalizer.apply(obs).unsqueeze(0)
        _ = self._unsynced_normalizer.step(obs)
        y, yhat = self._teacher_net(normalized), self._student_net(normalized)
        rewards_dict = {'rnd_intrinsic': tc.square(y-yhat).sum(dim=-1)}
        if isinstance(rew, dict):
            rewards_dict.update(rew)
        else:
            rewards_dict['extrinsic'] = rew
        return obs, rewards_dict, done, info

    def learn(self, obs_batch, **kwargs):
        normalized = self._synced_normalizer.apply(obs_batch)
        y, yhat = self._teacher_net(normalized), self._student_net(normalized)
        loss = tc.square(y-yhat).sum(dim=-1).mean(dim=0)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._sync_normalizers()
