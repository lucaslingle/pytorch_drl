"""
Policy heads.
"""

from typing import Mapping, Any, Type, Callable, Optional
import abc

import torch as tc

from drl.agents.heads.action_value_heads import (
    Head,
    DiscreteActionValueHead,
    SimpleDiscreteActionValueHead,
    DistributionalDiscreteActionValueHead)
from drl.agents.architectures.stateless.abstract import HeadEligibleArchitecture
from drl.agents.preprocessing.tabular import one_hot


class PolicyHead(Head, metaclass=abc.ABCMeta):
    """
    Abstract class for policy heads.
    """
    def __init__(self, num_features: int):
        """
        Args:
            num_features (int): Number of features.
        """
        super().__init__()
        self._num_features = num_features


class DiscretePolicyHead(PolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for discrete-action policy heads.
    """
    def __init__(self, num_features: int, num_actions: int):
        """
        Args:
            num_features (int): Number of features.
            num_actions (int): Number of actions.
        """
        PolicyHead.__init__(self, num_features)
        self._num_actions = num_actions


class ContinuousPolicyHead(PolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for continuous-action policy heads.
    """
    def __init__(self, num_features: int, action_dim: int):
        """
        Args:
            num_features (int): Number of features.
            action_dim (int): Action dimensionality.
        """
        PolicyHead.__init__(self, num_features)
        self._action_dim = action_dim


class CategoricalPolicyHead(DiscretePolicyHead, metaclass=abc.ABCMeta):
    """
    Categorically-distributed policy head.
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Mapping[str, Any],
            w_init: Optional[Callable[[tc.Tensor], None]],
            b_init: Optional[Callable[[tc.Tensor], None]],
            **kwargs: Mapping[str, Any]):
        """
        Args:
            num_features (int): Number of input features.
            num_actions (int): Number of actions.
            head_architecture_cls (Type[HeadEligibleArchitecture]): Class object
                for policy head architecture. Must be a derived class of
                HeadEligibleArchitecture.
            head_architecture_cls_args (Mapping[str, Any]): Keyword arguments
                for head architecture.
            w_init (Optional[Callable[[torch.Tensor], None]]): Weight initializer.
            b_init (Optional[Callable[[torch.Tensor], None]]): Bias initializer.
            **kwargs (Mapping[str, Any]): Keyword arguments.
        """
        DiscretePolicyHead.__init__(self, num_features, num_actions)
        self._policy_head = head_architecture_cls(
            input_dim=num_features,
            output_dim=num_actions,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def forward(
            self, features: tc.Tensor,
            **kwargs: Mapping[str, Any]) -> tc.distributions.Categorical:
        """
        Args:
            features (torch.Tensor): Torch tensor with shape
                [batch_size, num_features].
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            torch.distributions.Categorical: Torch distribution
                with batch shape [batch_size], and event shape [].
        """
        logits = self._policy_head(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class DiagonalGaussianPolicyHead(ContinuousPolicyHead, metaclass=abc.ABCMeta):
    """
    Diagonal Gaussian policy head.
    """
    def __init__(
            self,
            num_features: int,
            action_dim: int,
            head_architecture_cls: Type[HeadEligibleArchitecture],
            head_architecture_cls_args: Mapping[str, Any],
            w_init: Optional[Callable[[tc.Tensor], None]],
            b_init: Optional[Callable[[tc.Tensor], None]],
            **kwargs: Mapping[str, Any]):
        """
        Args:
            num_features (int): Number of input features.
            action_dim (int): Action dimensionality.
            head_architecture_cls (Type[HeadEligibleArchitecture]): Class object
                for policy head architecture. Must be a derived class of
                HeadEligibleArchitecture.
            head_architecture_cls_args (Mapping[str, Any]): Keyword arguments
                for head architecture.
            w_init (Optional[Callable[[torch.Tensor], None]]): Weight initializer.
            b_init (Optional[Callable[[torch.Tensor], None]]): Bias initializer.
            **kwargs (Mapping[str, Any]): Keyword arguments.
        """
        ContinuousPolicyHead.__init__(self, num_features, action_dim)
        self._policy_head = head_architecture_cls(
            input_dim=num_features,
            output_dim=action_dim * 2,
            w_init=w_init,
            b_init=b_init,
            **head_architecture_cls_args)

    def forward(
            self, features: tc.Tensor,
            **kwargs: Mapping[str, Any]) -> tc.distributions.Independent:
        """
        Args:
            features (torch.Tensor): Torch tensor with shape
                [batch_size, num_features].
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            torch.distributions.Independent: Torch distribution with
                batch shape [batch_size], and event shape [action_dim].
        """
        vec = self._policy_head(features)
        mu, logsigma = tc.chunk(vec, 2, dim=-1)
        dist = tc.distributions.Normal(loc=mu, scale=tc.exp(logsigma))
        dist = tc.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        return dist


class EpsilonGreedyCategoricalPolicyHead(DiscretePolicyHead):
    """
    Epsilon-greedy policy head for use in conjunction with an action-value
    function.
    """
    def __init__(
            self,
            action_value_head: DiscreteActionValueHead,
            **kwargs: Mapping[str, Any]):
        """
        Args:
            action_value_head (DiscreteActionValueHead):
                DiscreteActionValueHead instance.
            **kwargs (Mapping[str, Any]): Keyword arguments.
        """
        num_features = action_value_head.num_features
        num_actions = action_value_head.num_actions
        super().__init__(num_features, num_actions)
        self._action_value_head = action_value_head
        self._epsilon = None

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        assert 0. <= value <= 1.
        self._epsilon = value

    def forward(
            self, features: tc.Tensor,
            **kwargs: Mapping[str, Any]) -> tc.distributions.Categorical:
        """
        Args:
            features (torch.Tensor): Torch tensor with shape
                [batch_size, num_features].
            **kwargs (Mapping[str, Any]): Keyword arguments.

        Returns:
            torch.distributions.Categorical: Torch categorical distribution
                with batch shape [batch_size], and element shape [num_actions].
        """
        if isinstance(self._action_value_head, SimpleDiscreteActionValueHead):
            q_values = self._action_value_head(features)
        elif isinstance(self._action_value_head,
                        DistributionalDiscreteActionValueHead):
            q_logits = self._action_value_head(features)
            q_values = self._action_value_head.logits_to_mean(q_logits)
        else:
            msg = "Action-value head class not supported."
            raise NotImplementedError(msg)
        num_actions = q_values.shape[-1]
        greedy_action = tc.argmax(q_values, dim=-1)
        greedy_policy = one_hot(greedy_action, depth=num_actions)
        uniform_policy = tc.ones_like(q_values)
        uniform_policy /= uniform_policy.sum(dim=-1, keepdim=True)
        probs = (1 - self.epsilon) * greedy_policy + \
            self.epsilon * uniform_policy
        dist = tc.distributions.Categorical(probs=probs)
        return dist
