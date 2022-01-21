from typing import Dict, List, Any
import importlib

import torch as tc
import gym

from drl.agents.preprocessing import Preprocessing
from drl.agents.architectures import Architecture
from drl.agents.heads import (
    Head, DiscreteActionValueHead, EpsilonGreedyCategoricalPolicyHead
)


def get_preprocessing(cls_name, cls_args):
    module = importlib.import_module('drl.agents.preprocessing')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_preprocessings(**preprocessing_spec: Dict[str, Dict[str, Any]]):
    preprocessing_stack = list()
    for cls_name, cls_args in preprocessing_spec.items():
        preprocessing = get_preprocessing(
            cls_name=cls_name, cls_args=cls_args)
        preprocessing_stack.append(preprocessing)
    return preprocessing_stack


def get_architecture(cls_name, cls_args):
    module = importlib.import_module('drl.agents.architectures')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_predictor(cls_name, cls_args):
    module = importlib.import_module('drl.agents.heads')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_epsilon_sched(rank, cls_name, cls_args):
    module = importlib.import_module('drl.agents.schedules')
    cls = getattr(module, cls_name)
    return cls(rank=rank, **cls_args)


def get_predictors(rank, env, **predictors_spec: Dict[str, Dict[str, Any]]):
    predictors = dict()

    if 'epsilon_schedule' in predictors_spec:
        epsilon_schedule_spec = predictors_spec.pop('epsilon_schedule')
    else:
        epsilon_schedule_spec = None

    for key, spec in predictors_spec.items():
        # infer number of actions or action dimensionality.
        if key == 'policy' or key.startswith('action_value'):
            if isinstance(env.action_space, gym.spaces.Discrete):
                spec['cls_args'].update({'num_actions': env.action_space.n})
            elif isinstance(env.action_space, gym.spaces.Box):
                spec['cls_args'].update({'action_dim': env.action_space.shape[0]})
            else:
                msg = "Unknown action space."
                raise TypeError(msg)

        # create and add the predictor.
        predictor = get_predictor(**spec)
        predictors[key] = predictor

        # add additional predictor for epsilon-greedy policy in DQN.
        if isinstance(predictor, DiscreteActionValueHead):
            if epsilon_schedule_spec is None:
                msg = "Required predictor epsilon_schedule missing from config."
                raise ValueError(msg)
            epsilon_schedule = get_epsilon_sched(
                rank=rank, **epsilon_schedule_spec)
            policy_predictor = EpsilonGreedyCategoricalPolicyHead(
                action_value_head=predictor, epsilon_schedule=epsilon_schedule)
            predictors['policy'] = policy_predictor

    return predictors


class Agent(tc.nn.Module):
    def __init__(
            self,
            preprocessing: List[Preprocessing],
            architecture: Architecture,
            predictors: Dict[str, Head],
            detach_input: bool = True
    ):
        super().__init__()
        self._preprocessing = tc.nn.Sequential(*preprocessing)
        self._architecture = architecture
        self._predictors = tc.nn.ModuleDict(predictors)
        self._detach_input = detach_input

    @property
    def keys(self):
        return self._predictors.keys()

    def forward(self, x, predict, **kwargs):
        """
        Args:
            x: Batch of observations
            predict: Names of predictors to apply.
            kwargs: Keyword arguments for predictors.
        Returns:
            Dictionary of predictions.
        """
        if self._detach_input:
            x = x.detach()
        preproc = self._preprocessing(x)
        features = self._architecture(preproc)
        predictions = {
            key: self._predictors[key](features, **kwargs) for key in predict
        }
        return predictions
