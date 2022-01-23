from typing import Mapping, Any, Type, Callable, Dict
import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture


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

    Reference:
        M. Bellemare et al., 2017 -
            'A Distributional Perspective on Reinforcement Learning'.
    """
    def __init__(self, vmin, vmax, num_bins):
        """
        Args:
            vmin: Minimum return value.
            vmax: Maximum return value.
            num_bins: Number of bins for distributional value learning.
        """
        super().__init__()
        self._vmin = vmin
        self._vmax = vmax
        self._num_bins = num_bins

    def returns_to_bin_ids(self, returns):
        returns = tc.clip(returns, self._vmin, self._vmax)
        bin_width = (self._vmax - self._vmin) / self._num_bins
        bin_edges = self._vmin + bin_width * tc.arange(self._num_bins+1).float()
        indices = tc.bucketize(returns, bin_edges)
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


class SimpleDiscreteActionValueHead(
    SimpleActionValueHead, DiscreteActionValueHead):
    """
    Simple discrete-action action-value prediction head.

    References:
        V. Mnih et al., 2015 -
            'Human Level Control through Deep Reinforcement Learning'
        Z. Wang et al., 2016 -
            'Dueling Network Architectures for Deep Reinforcement Learning'
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Dict[str, Any],
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            num_features: Number of input features.
            num_actions: Number of actions.
            head_architecture_cls: Class name for policy head architecture.
                Must be a derived class of HeadEligibleArchitecture.
            head_architecture_cls_args: Keyword arguments for head architecture.
            w_init: Weight initializer.
            b_init: Bias initializer.
            **kwargs: Keyword arguments.
        """
        SimpleActionValueHead.__init__(self)
        DiscreteActionValueHead.__init__(self, num_actions)
        self._action_value_head = head_architecture_cls(
            input_dim=num_features,
            output_dim=num_actions,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def forward(self, features, **kwargs):
        qpreds = self._action_value_head(features)
        return qpreds


class SimpleContinuousActionValueHead(
    SimpleActionValueHead, ContinuousActionValueHead):
    """
    Simple continuous-action action-value prediction head.

    Reference:
        T. Lillicrap et al., 2015 -
            'Continuous Control with Deep Reinforcement Learning'.
    """
    def __init__(
            self,
            num_features: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Dict[str, Any],
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            num_features: Number of input features.
            head_architecture_cls: Class name for policy head architecture.
                Must be a derived class of HeadEligibleArchitecture.
            head_architecture_cls_args: Keyword arguments for head architecture.
            w_init: Weight initializer.
            b_init: Bias initializer.
            **kwargs: Keyword arguments.
        """
        SimpleActionValueHead.__init__(self)
        ContinuousActionValueHead.__init__(self)
        self._action_value_head = head_architecture_cls(
            input_dim=num_features,
            output_dim=1,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def forward(self, features, **kwargs):
        qpred = self._action_value_head(features).squeeze(-1)
        return qpred


class DistributionalDiscreteActionValueHead(
    DistributionalActionValueHead, DiscreteActionValueHead):
    """
    Distributional discrete-action action-value prediction head.

    Reference:
        M. Bellemare et al., 2017 -
            'A Distributional Perspective on Reinforcement Learning'.
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Dict[str, Any],
            w_init: Callable[[tc.Tensor], None],
            b_init: Callable[[tc.Tensor], None],
            vmin: float,
            vmax: float,
            num_bins: int,
            **kwargs: Mapping[str, Any]
    ):
        """
        Args:
            num_features: Number of input features.
            num_actions: Number of actions.
            head_architecture_cls: Class name for policy head architecture.
                Must be a derived class of HeadEligibleArchitecture.
            head_architecture_cls_args: Keyword arguments for head architecture.
            w_init: Weight initializer.
            b_init: Bias initializer.
            vmin: Minimum return value.
            vmax: Maximum return value.
            num_bins: Number of bins for distributional value learning.
            **kwargs: Keyword arguments.
        """
        DistributionalActionValueHead.__init__(self, vmin, vmax, num_bins)
        DiscreteActionValueHead.__init__(self, num_actions)
        self._action_value_head = head_architecture_cls(
            input_dim=num_features,
            output_dim=num_actions * num_bins,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def logits_to_mean(self, q_value_logits: tc.Tensor) -> tc.Tensor:
        """
        Args:
            q_value_logits: Torch tensor of shape
                [batch_size, num_actions, num_bins]
                containing action-value logits.

        Returns:
            Torch tensor of shape [batch_size, num_actions] containing the mean
            q-value predicted for each action.
        """
        bin_width = (self._vmax - self._vmin) / self._num_bins
        bin_midpoints = self._vmin + 0.5 * bin_width + \
            bin_width * tc.arange(self._num_bins).float()
        bin_midpoints = bin_midpoints.view(1, 1, self._num_bins)
        value_dists = tc.nn.functional.softmax(dim=-1)(q_value_logits)
        q_value_means = (value_dists * bin_midpoints).sum(dim=-1)
        return q_value_means

    def forward(self, features, **kwargs):
        """
        Args:
            features: Torch tensor with shape [batch_size, num_features].
            **kwargs: Keyword arguments.

        Returns:
            Torch tensor with shappe [batch_size, num_actions, num_bins],
            containing the logits of the estimated state-action-conditional
            value distribution.
        """
        q_value_logits_flat = self._action_value_head(features)
        q_value_logits = q_value_logits_flat.reshape(
            -1, self._num_actions, self._num_bins)
        return q_value_logits
