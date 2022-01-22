from typing import Tuple, Mapping, Any
import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.integration import get_architecture


class ActionValueHead(Head, metaclass=abc.ABCMeta):
    """
    Abstract class for action-value prediction heads.
    """


class SimpleActionValueHead(ActionValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for simple action-value prediction heads
    (as opposed to distributional).
    """


class DistributionalActionValueHead(ActionValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for distributional action-value prediction heads.

    Reference: Bellemare et al., 2017 -
        'A Distributional Perspective on Reinforcement Learning'.
    """
    def __init__(self, vmin, vmax, num_bins):
        super().__init__()
        self._vmin = vmin
        self._vmax = vmax
        self._num_bins = num_bins

    def returns_to_bin_ids(self, returns):
        returns = tc.clip(returns, self._vmin, self._vmax)
        bin_width = (self._vmax - self._vmin) / self._num_bins
        bin_values = self._vmin + bin_width * tc.arange(self._num_bins).float()
        indices = tc.bucketize(returns, bin_values)
        return indices


class DiscreteActionValueHead(ActionValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for action-value prediction heads
    with outputted predictions for each action.
    """
    def __init__(self, num_actions):
        super().__init__()
        self._num_actions = num_actions


class ContinuousActionValueHead(ActionValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for action-value prediction heads
    with predictions for only the inputted action.
    """


class SimpleDiscreteActionValueHead(SimpleActionValueHead, DiscreteActionValueHead):
    """
    Simple discrete-action action-value prediction head.
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            architecture_cls_name: str,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]]
    ):
        SimpleActionValueHead.__init__(self)
        DiscreteActionValueHead.__init__(self, num_actions)
        self._action_value_head = get_architecture(
            cls_name=architecture_cls_name,
            cls_args={
                'input_dim': num_features,
                'output_dim': num_actions,
                'w_init_spec': w_init_spec,
                'b_init_spec': b_init_spec
            })

    def forward(self, features, **kwargs):
        q_values = self._action_value_head(features)
        return q_values


class SimpleContinuousActionValueHead(SimpleActionValueHead, ContinuousActionValueHead):
    """
    Simple continuous-action action-value prediction head.
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            architecture_cls_name: str,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]]
    ):
        SimpleActionValueHead.__init__(self)
        ContinuousActionValueHead.__init__(self)
        self._action_value_head = get_architecture(
            cls_name=architecture_cls_name,
            cls_args={
                'input_dim': num_features,
                'output_dim': num_actions,
                'w_init_spec': w_init_spec,
                'b_init_spec': b_init_spec
            })

    def forward(self, features, **kwargs):
        q_value = self._action_value_head(features).squeeze(-1)
        return q_value


class DistributionalDiscreteActionValueHead(DistributionalActionValueHead, DiscreteActionValueHead):
    """
    Distributional discrete-action action-value prediction head.
    Reference: Bellemare et al., 2017.
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            architecture_cls_name: str,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]],
            vmin: float,
            vmax: float,
            num_bins: int
    ):
        """
        Args:
            num_features: Number of input features.
            num_actions: Number of actions.
            architecture_cls_name: Class name for policy head architecture.
                Must be a derived class of StatelessArchitecture.
            w_init_spec: Tuple containing weight initializer name and kwargs.
            b_init_spec: Tuple containing bias initializer name and kwargs.
            vmin: Minimum return value.
            vmax: Maximum return value.
            num_bins: Number of bins for distributional value learning.
        """
        DistributionalActionValueHead.__init__(self, vmin, vmax, num_bins)
        DiscreteActionValueHead.__init__(self, num_actions)
        self._action_value_head = get_architecture(
            cls_name=architecture_cls_name,
            cls_args={
                'input_dim': num_features,
                'output_dim': num_actions,
                'w_init_spec': w_init_spec,
                'b_init_spec': b_init_spec
            })

    def forward(self, features, **kwargs):
        q_value_logits_flat = self._action_value_head(features)
        q_value_logits = q_value_logits_flat.reshape(
            -1, self._num_actions, self._num_bins)

        bin_width = (self._vmax - self._vmin) / self._num_bins
        bin_values = self._vmin + bin_width * tc.arange(self._num_bins).float()
        bin_values = bin_values.view(1, 1, self._num_bins)
        q_value_means = (tc.nn.Softmax(dim=-1)(q_value_logits) * bin_values).sum(dim=-1)
        return {
            "q_value_logits": q_value_logits,
            "q_value_means": q_value_means
        }
