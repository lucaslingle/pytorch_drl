import torch as tc

from drl.envs.wrappers.stateful.abstract import (
    StatefulWrapper, TrainableWrapper
)
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.utils.optim_util import get_optimizer


class TeacherNetwork(tc.nn.Module):
    """
    Teacher network for Random Network Distillation.
    Assumes input image has dimension 84x84x4.
    """
    def __init__(self, data_shape):
        super().__init__()
        self._data_shape = data_shape
        # todo(lucaslingle): finish this.


class StudentNetwork(tc.nn.Module):
    """
    Student network for Random Network Distillation.
    Assumes input image has dimension 84x84x4.
    """
    def __init__(self, data_shape):
        super().__init__()
        self._data_shape = data_shape
        # todo(lucaslingle): finish this.


class RandomNetworkDistillationWrapper(TrainableWrapper):
    def __init__(self, env, rnd_optimizer_cls_name, rnd_optimizer_args):
        super().__init__(env)
        self._data_shape = [84, 84, 4]
        self._normalizer = Normalizer(self._data_shape, -5, 5)
        self._teacher_net = TeacherNetwork(self._data_shape)
        self._student_net = StudentNetwork(self._data_shape)
        self._optimizer = get_optimizer(
            rnd_optimizer_cls_name, rnd_optimizer_args)
        # todo(lucaslingle): Normalizer statistics are local,
        #  so updates will disagree in distributed case. How to fix this?

    def get_checkpointables(self):
        checkpointables = dict()
        if isinstance(self.env, StatefulWrapper):
            # todo(lucaslingle): This wont work if there are
            #  non-TrainableWrappers between TrainableWrappers. How to fix?
            checkpointables.update(self.env.get_checkpointables())
        checkpointables.update({
            'rnd_observation_normalizer': self._normalizer,
            'rnd_teacher_net': self._teacher_net,
            'rnd_student_net': self._student_net,
            'rnd_optimizer': self._optimizer
        })
        return checkpointables

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        normalized = self._normalizer(obs)
        y, yhat = self._teacher_net(normalized), self._student_net(normalized)
        rewards_dict = {'rnd_intrinsic': tc.square(y-yhat).sum(dim=-1)}
        if isinstance(rew, dict):
            rewards_dict.update(rew)
        else:
            rewards_dict['extrinsic'] = rew
        return obs, rewards_dict, done, info
