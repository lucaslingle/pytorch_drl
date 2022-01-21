import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.common import normc_init_


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
    Reference: Bellemare et al., 2017.
    """
    def returns_to_bin_ids(self, returns):
        returns = tc.clip(returns, self._vmin, self._vmax)
        bin_width = (self._vmax - self._vmin) / self._num_bins
        bin_values = self._vmin + bin_width * tc.arange(self._num_bins).float()
        indices = tc.bucketize(returns, bin_values)
        return indices


class CategoricalActionValueHead(ActionValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for action-value prediction heads
    with outputted predictions for each action.
    """


class IngressActionValueHead(ActionValueHead, metaclass=abc.ABCMeta):
    """
    Abstract class for action-value prediction heads
    with predictions for only the inputted action.
    """


class SimpleCategoricalActionValueHead(
    SimpleActionValueHead, CategoricalActionValueHead, metaclass=abc.ABCMeta
):
    """
    Abstract class for simple categorical-action action-value heads.
    """
    def forward(self, features, **kwargs):
        q_values = self._action_value_head(features)
        return q_values


class SimpleIngressActionValueHead(
        SimpleActionValueHead, IngressActionValueHead, metaclass=abc.ABCMeta
    ):
    """
    Abstract class for simple ingress-action action-value heads.
    Useful primarily for off-policy continuous control.
    """
    def forward(self, features, **kwargs):
        q_value = self._action_value_head(features).squeeze(-1)
        return q_value


class DistributionalCategoricalActionValueHead(
    DistributionalActionValueHead, CategoricalActionValueHead, metaclass=abc.ABCMeta
):
    """
    Abstract class for distributional categorical-action action-value heads.
    """
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


class DistributionalIngressActionValueHead(
    DistributionalActionValueHead, IngressActionValueHead, metaclass=abc.ABCMeta
):
    """
    Abstract class for distributional ingress-based action-value heads.
    """
    def forward(self, features, **kwargs):
        q_value_logits_flat = self._action_value_head(features)
        q_value_logits = q_value_logits_flat.reshape(-1, self._num_bins)

        bin_width = (self._vmax - self._vmin) / self._num_bins
        bin_values = self._vmin + bin_width * tc.arange(self._num_bins).float()
        bin_values = bin_values.view(1, self._num_bins)
        q_value_means = (tc.nn.Softmax(dim=-1)(q_value_logits) * bin_values).sum(dim=-1)
        return {
            "q_value_logits": q_value_logits,
            "q_value_means": q_value_means
        }


class LinearSimpleCategoricalActionValueHead(SimpleCategoricalActionValueHead):
    """
    Simple action-value prediction head for categorical actions,
    using a linear projection of state features.
    """
    def __init__(
            self,
            num_features,
            num_actions,
            ortho_init,
            ortho_gain=0.01,
            **kwargs
    ):
        """
        Args:
            num_features: Number of features.
            num_actions: Number of actions.
            ortho_init: Use orthogonal initialization?
            ortho_gain: Gain parameter for orthogonal initialization.
                Ignored if ortho_init is False.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, num_actions)
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._action_value_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._action_value_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._value_head.bias)


class LinearSimpleIngressActionValueHead(SimpleIngressActionValueHead):
    """
    Simple action-value prediction head,
    using a linear projection of state-action features.

    Useful primarily for off-policy continuous control.
    """
    def __init__(self, num_features, ortho_init, ortho_gain=0.01, **kwargs):
        """
        Args:
            num_features: Number of features.
            ortho_init: Use orthogonal initialization?
            ortho_gain: Gain parameter for orthogonal initialization.
                Ignored if ortho_init is False.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, 1)
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._action_value_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._action_value_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._value_head.bias)


class LinearDistributionalCategoricalActionValueHead(
    DistributionalCategoricalActionValueHead
):
    """
    Distributional action-value prediction head for categorical actions,
    using a linear projection of state features.
    """
    def __init__(
            self,
            num_features,
            num_actions,
            vmin,
            vmax,
            num_bins,
            ortho_init,
            ortho_gain=0.01,
            **kwargs
    ):
        """
        Args:
            num_features: Number of features.
            num_actions: Number of actions.
            vmin: Minimum value for discounted total returns.
            vmax: Maximum value for discounted total returns.
            num_bins: Number of bins for returns.
            ortho_init: Use orthogonal initialization?
            ortho_gain: Gain parameter for orthogonal initialization.
                Ignored if ortho_init is False.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._action_value_head = tc.nn.Linear(
            num_features, num_actions * num_bins)
        self._num_features = num_features
        self._num_actions = num_actions
        self._vmin = vmin
        self._vmax = vmax
        self._num_bins = num_bins
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._action_value_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._action_value_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._value_head.bias)


class LinearDistributionalIngressActionValueHead(
    DistributionalIngressActionValueHead
):
    """
    Distributional action-value prediction head,
    using a linear projection of state-action features.

    Useful primarily for off-policy continuous control.
    """
    def __init__(
            self,
            num_features,
            vmin,
            vmax,
            num_bins,
            ortho_init,
            ortho_gain=0.01,
            **kwargs):
        """
        Args:
            num_features: Number of features.
            vmin: Minimum value for discounted total returns.
            vmax: Maximum value for discounted total returns.
            num_bins: Number of bins for returns.
            ortho_init: Use orthogonal initialization?
            ortho_gain: Gain parameter for orthogonal initialization.
                Ignored if ortho_init is False.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._action_value_head = tc.nn.Linear(num_features, num_bins)
        self._num_features = num_features
        self._vmin = vmin
        self._vmax = vmax
        self._num_bins = num_bins
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._action_value_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._action_value_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._value_head.bias)