import abc

import torch as tc

from drl.agents.heads.abstract import Head
from drl.agents.architectures.common import normc_init_
from drl.agents.preprocessing.tabular import one_hot


class PolicyHead(Head, metaclass=abc.ABCMeta):
    """
    Abstract class for policy heads.
    """


class DiscretePolicyHead(PolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for discrete-action policy heads.
    """


class ContinuousPolicyHead(PolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for continuous-action policy heads.
    """


class CategoricalPolicyHead(DiscretePolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for simple categorical policy heads.

    It is distinct from other discrete-action policy heads,
    like discrete autoregressive ones that may be used
    in some complex environments.
    """
    def forward(self, features, **kwargs):
        """
        Args:
            features: Torch tensor with shape [batch_size, num_features].
            **kwargs: Keyword arguments.

        Returns:
            Torch categorical distribution with batch shape [batch_size],
            and element shape [action_dim].
        """
        logits = self._policy_head(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class DiagonalGaussianPolicyHead(ContinuousPolicyHead, metaclass=abc.ABCMeta):
    """
    Abstract class for simple diagonal Gaussian policy heads.
    """
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


class EpsilonGreedyCategoricalPolicyHead(CategoricalPolicyHead):
    """
    Epsilon-greedy policy head for use in conjunction with an action-value
    function.
    """
    def __init__(self, action_value_head, epsilon_schedule, **kwargs):
        """
        Args:
            action_value_head: ActionValueHead instance.
            epsilon_schedule: EpsilonSchedule instance.
            **kwargs:
        """
        super().__init__()
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


class GreedyCategoricalPolicyHead(CategoricalPolicyHead):
    """
    Greedy policy head for use in conjunction with an action-value function.
    """
    def __init__(self, action_value_head, **kwargs):
        """
        Args:
            action_value_head: ActionValueHead instance.
            **kwargs:
        """
        super().__init__()
        self._action_value_head = action_value_head

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
        q_values = self._action_value_head(features)
        num_actions = q_values.shape[-1]
        greedy_action = tc.argmax(q_values, dim=-1)
        greedy_policy = one_hot(greedy_action, depth=num_actions)
        dist = tc.distributions.Categorical(probs=greedy_policy)
        return dist


class LinearCategoricalPolicyHead(CategoricalPolicyHead):
    """
    Categorical policy head with a linear projection to the action logits.
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
        self._policy_head = tc.nn.Linear(num_features, num_actions)
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._policy_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._policy_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._policy_head.bias)


class LinearDiagonalGaussianPolicyHead(DiagonalGaussianPolicyHead):
    """
    Diagonal Gaussian policy head with a linear projection to the location
    and log-scale parameters.
    """
    def __init__(
            self,
            num_features,
            action_dim,
            ortho_init,
            ortho_gain=0.01,
            **kwargs
    ):
        """
        Args:
            num_features: Number of features.
            action_dim: Action space dimensionality.
            ortho_init: Use orthogonal initialization?
            ortho_gain: Gain parameter for orthogonal initialization.
                Ignored if ortho_init is False.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self._policy_head = tc.nn.Linear(num_features, 2*action_dim)
        self._ortho_init = ortho_init
        self._ortho_gain = ortho_gain
        self._init_weights()

    def _init_weights(self):
        if self._ortho_init:
            tc.nn.init.orthogonal_(
                self._policy_head.weight, gain=self._ortho_gain)
        else:
            normc_init_(self._policy_head.weight, gain=0.01)
        tc.nn.init.zeros_(self._policy_head.bias)


