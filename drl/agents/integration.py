from typing import Dict, List, Any, Mapping, Union, Type, Tuple
import importlib

import torch as tc
import gym

from drl.agents.preprocessing import Preprocessing
from drl.agents.architectures import Architecture
from drl.agents.architectures.stateless.abstract import StatelessArchitecture
from drl.agents.architectures.initializers import get_initializer
from drl.agents.heads import (
    Head,
    DiscreteActionValueHead,
    EpsilonGreedyCategoricalPolicyHead,
    DiscretePolicyHead,
    ContinuousPolicyHead
)
from drl.envs.wrappers.stateless.abstract import Wrapper


def get_preprocessing(
        cls_name: str, cls_args: Mapping[str, Any]) -> Preprocessing:
    """
    Args:
        cls_name: Name of a derived class of Preprocessing.
        cls_args: Arguments in the signature of the class constructor.

    Returns:
        Instantiated class.
    """
    module = importlib.import_module('drl.agents.preprocessing')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_preprocessings(
        **preprocessing_spec: Mapping[str, Mapping[str, Any]]
) -> List[Preprocessing]:
    """
    Args:
        **preprocessing_spec: Variable length dictionary of preprocessing specs.
            Each preprocessing spec is keyed by a class name,
            which should be a derived class of Preprocessing.
            Each preprocessing spec's key maps to a value, which is a dictionary
            of arguments passed to the constructor of that class.

    Returns:
        List of instantiated Preprocessing subclasses.
    """
    preprocessing_stack = list()
    for cls_name, cls_args in preprocessing_spec.items():
        preprocessing = get_preprocessing(
            cls_name=cls_name, cls_args=cls_args)
        preprocessing_stack.append(preprocessing)
    return preprocessing_stack


def get_architecturec_cls(cls_name: str) -> Type[Architecture]:
    """
    Args:
        cls_name: Class name.

    Returns:
        Class object.
    """
    module = importlib.import_module('drl.agents.architectures')
    cls = getattr(module, cls_name)
    return cls


def get_architecture(
        cls_name: str,
        cls_args: Dict[str, Any],
        w_init_spec: Tuple[str, Dict[str, Any]],
        b_init_spec: Tuple[str, Dict[str, Any]]
) -> Architecture:
    """
    Args:
        cls_name: Name of a derived class of Architecture.
        cls_args: Arguments in the signature of the class constructor.
        w_init_spec: Tuple containing weight initializer name and args.
        b_init_spec: Tuple containing bias initializer name and args.

    Returns:
        Instantiated class.
    """
    cls = get_architecturec_cls(cls_name)
    args = {
        **cls_args,
        'w_init': get_initializer(w_init_spec),
        'b_init': get_initializer(b_init_spec)
    }
    return cls(**args)


def get_predictor(
        cls_name: str,
        cls_args: Dict[str, Any],
        head_architecture_cls_name: str,
        head_architecture_cls_args: Dict[str, Any],
        w_init_spec: Tuple[str, Dict[str, Any]],
        b_init_spec: Tuple[str, Dict[str, Any]]
) -> Head:
    """
    Args:
        cls_name: Head class name.
        cls_args: Head class constructor arguments.
            Should contain at least 'num_features'
            and either 'num_actions' or 'action_dim'.
        head_architecture_cls_name: Class name for head architecture.
            Should correspond to a derived class of HeadEligibleArchitecture.
        head_architecture_cls_args: Class arguments for head architecture.
        w_init_spec: Tuple containing weight initializer name and args.
        b_init_spec: Tuple containing bias initializer name and args.

    Returns:
        Instantiated Head subclass.
    """
    assert 'num_features' in cls_args
    module = importlib.import_module('drl.agents.heads')
    head_cls = getattr(module, cls_name)
    if issubclass(head_cls, (DiscretePolicyHead, DiscreteActionValueHead)):
        assert 'num_actions' in cls_args
    if issubclass(head_cls, ContinuousPolicyHead):
        assert 'action_dim' in cls_args

    args = {
        **cls_args,
        'head_architecture_cls': get_architecturec_cls(
            head_architecture_cls_name),
        'head_architecture_cls_args': head_architecture_cls_args,
        'w_init': get_initializer(w_init_spec),
        'b_init': get_initializer(b_init_spec)
    }
    return head_cls(**args)


def get_predictors(
        env: Union[gym.core.Env, Wrapper],
        **predictors_spec: Mapping[str, Dict[str, Any]]) -> Dict[str, Head]:
    """
    Args:
        env: OpenAI gym environment instance or wrapped environment.
        **predictors_spec: Variable length dictionary of predictor specs.
            Each predictor spec is keyed by a prediction key.
            Each predictor spec's key maps to a value, which is a dictionary
            with keys 'cls_name' and 'cls_args'.
            These keys map to values for names of derived classes of Head,
            and to dictionaries of arguments to be passed to each class'
            constructor.

    Returns:
        Dictionary of predictors keyed by name.
    """
    predictors = dict()
    for key, spec in predictors_spec.items():
        # infer number of actions or action dimensionality.
        if key == 'policy' or key.startswith('action_value_'):
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
