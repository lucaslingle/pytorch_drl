from typing import Dict, List, Any
import importlib

import torch as tc
import gym

from drl.agents.preprocessing import Preprocessing
from drl.agents.architectures.stateless.abstract import StatelessArchitecture
from drl.agents.heads import (
    Head,
    DiscreteActionValueHead,
    EpsilonGreedyCategoricalPolicyHead
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


def get_predictors(env, **predictors_spec: Dict[str, Dict[str, Any]]):
    predictors = dict()
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
            eps_greedy_policy_predictor = EpsilonGreedyCategoricalPolicyHead(
                action_value_head=predictor)
            predictors['policy'] = eps_greedy_policy_predictor
    return predictors


class Agent(tc.nn.Module):
    def __init__(
            self,
            preprocessing: List[Preprocessing],
            architecture: StatelessArchitecture,
            predictors: Dict[str, Head],
            detach_input: bool = True
    ):
        """
        Args:
            preprocessing: List of preprocessing operations.
            architecture: StatelessArchitecture instance.
            predictors: Dictionary of prediction heads, keyed by prediction name.
            detach_input: Detach input or not? Default: True.
                This is a no-op if used with external observation tensors,
                since these are outside the model.parameters() passed to the
                optimizer and moreover default to have requires_grad=False.
        """
        super().__init__()
        self._preprocessing = tc.nn.Sequential(*preprocessing)
        self._architecture = architecture
        self._predictors = tc.nn.ModuleDict(predictors)
        self._detach_input = detach_input
        self._check_predict_keys()

    def _check_predict_keys(self):
        for key in self.keys:
            if key == 'policy':
                continue
            if key.startswith('value_'):
                continue
            if key.startswith('action_value_'):
                continue
            raise ValueError(f"Prediction key {key} not supported.")

    @property
    def keys(self):
        return self._predictors.keys()

    def forward(self, observations, predict, **kwargs):
        """
        Args:
            observations: Batch of observations
            predict: Names of predictors to apply.
            kwargs: Keyword arguments.
        Returns:
            Dictionary of predictions.
        """
        if self._detach_input:
            observations = observations.detach()
        preprocessed = self._preprocessing(observations)
        features = self._architecture(preprocessed, **kwargs)
        predictions = {
            key: self._predictors[key](features, **kwargs) for key in predict
        }
        return predictions
