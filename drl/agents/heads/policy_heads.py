from typing import Tuple, Mapping, Any
import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.preprocessing.tabular import one_hot
from drl.agents.integration import get_architecture


class PolicyHead(Head, metaclass=abc.ABCMeta):
    """
    Abstract class for policy heads.
    """
    def __init__(self, num_features):
        super().__init__()
        self._num_features = num_features


class DiscretePolicyHead(PolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for discrete-action policy heads.
    """
    def __init__(self, num_features, num_actions):
        super().__init__(num_features)
        self._num_actions = num_actions


class ContinuousPolicyHead(PolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for continuous-action policy heads.
    """
    def __init__(self, num_features, action_dim):
        super().__init__(num_features)
        self._action_dim = action_dim


# default for ppo was architecture_cls_name=Linear, w_init=('normc', {'gain': 0.01}), b_init=('zeros_', {})
class CategoricalPolicyHead(DiscretePolicyHead, metaclass=abc.ABCMeta):
    """
    Categorically-distributed policy head.
    """
    def __init__(
            self,
            num_features: int,
            num_actions: int,
            architecture_cls_name: str,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]]
    ):
        """
        Args:
            num_features: Number of input features.
            num_actions: Number of actions.
            architecture_cls_name: Class name for policy head architecture.
                Must be a derived class of StatelessArchitecture.
            w_init_spec: Tuple containing weight initializer name and kwargs.
            b_init_spec: Tuple containing bias initializer name and kwargs.
        """
        super().__init__(num_features, num_actions)
        self._policy_head = get_architecture(
            cls_name=architecture_cls_name,
            cls_args={
                'input_dim': num_features,
                'output_dim': num_actions,
                'w_init_spec': w_init_spec,
                'b_init_spec': b_init_spec
            })

    def forward(self, features, **kwargs):
        """
        Args:
            features: Torch tensor with shape [batch_size, num_features].
            **kwargs: Keyword arguments.

        Returns:
            Torch categorical distribution with batch shape [batch_size],
            and element shape [num_actions].
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
            architecture_cls_name: str,
            w_init_spec: Tuple[str, Mapping[str, Any]],
            b_init_spec: Tuple[str, Mapping[str, Any]]
    ):
        """
        Args:
            num_features: Number of input features.
            action_dim: Action space dimensionality.
            architecture_cls_name: Class name for policy head architecture.
                Must be a derived class of StatelessArchitecture.
            w_init_spec: Tuple containing weight initializer name and kwargs.
            b_init_spec: Tuple containing bias initializer name and kwargs.
        """
        super().__init__(num_features, action_dim)
        self._policy_head = get_architecture(
            cls_name=architecture_cls_name,
            cls_args={
                'input_dim': num_features,
                'output_dim': action_dim * 2,
                'w_init_spec': w_init_spec,
                'b_init_spec': b_init_spec
            })

    def forward(self, features, **kwargs):
        """
        Args:
            features: Torch tensor with shape [batch_size, num_features].
            **kwargs: Keyword arguments.

        Returns:
            Torch normal distribution with batch shape [batch_size],
            and element shape [action_dim].
        """
        vec = self._policy_head(features)
        mu, logsigma = tc.chunk(vec, 2, dim=-1)
        dist = tc.distributions.Normal(loc=mu, scale=tc.exp(logsigma))
        return dist


class EpsilonGreedyCategoricalPolicyHead(DiscretePolicyHead):
    """
    Epsilon-greedy policy head for use in conjunction with an action-value
    function.
    """
    def __init__(self, action_value_head, epsilon_schedule, **kwargs):
        """
        Args:
            action_value_head: ActionValueHead instance.
            epsilon_schedule: EpsilonSchedule instance.
            **kwargs: Keyword arguments.
        """
        num_features = action_value_head.input_dim
        num_actions = action_value_head.output_dim
        super().__init__(num_features, num_actions)
        self._action_value_head = action_value_head
        self._epsilon_schedule = epsilon_schedule

    def forward(self, features, step, **kwargs):
        """
        Args:
            features: Torch tensor with shape [batch_size, num_features].
            step: Whether to step the epsilon scheduler on this forward call.
            **kwargs: Keyword arguments.

        Returns:
            Torch categorical distribution with batch shape [batch_size],
            and element shape [action_dim].
        """
        epsilon = self._epsilon_schedule.value
        if step:
            self._epsilon_schedule.step()
        q_values = self._action_value_head(features)
        num_actions = q_values.shape[-1]
        greedy_action = tc.argmax(q_values, dim=-1)
        greedy_policy = one_hot(greedy_action, depth=num_actions)
        uniform_policy = tc.ones_like(q_values)
        uniform_policy /= uniform_policy.sum(dim=-1, keepdim=True)
        probs = (1-epsilon) * greedy_policy + epsilon * uniform_policy
        dist = tc.distributions.Categorical(probs=probs)
        return dist
