from typing import Mapping, Any, Dict, Union
import abc
import importlib

import torch as tc


def extract_reward_name(prediction_key: str) -> str:
    """
    Extracts reward name from a prediction key.

    Args:
        prediction_key (str): Prediction key.
            Must start with 'value_' or 'action_value_'.

    Returns:
        str: Reward name.
    """
    prefixes = ['value_', 'action_value_']
    for prefix in prefixes:
        if prediction_key.startswith(prefix):
            return prediction_key[len(prefix):]
    raise ValueError("Unrecognized predictor name.")


def get_credit_assignment_op(
        cls_name: str,
        cls_args: Mapping[str, Any]
) -> 'CreditAssignmentOp':
    """
    Creates a credit assignment op from class name and args.

    Args:
        cls_name (str): Class name for a derived class of `CreditAssignmentOp`.
        cls_args (Mapping[str, Any]): Arguments for class constructor.

    Returns:
        CreditAssignmentOp: Instantiated credit assignment op.
    """
    module = importlib.import_module('drl.algos.common.credit_assignment')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_credit_assignment_ops(
        seg_len: int,
        extra_steps: int,
        credit_assignment_spec: Mapping[str, Mapping[str, Union[str, Mapping[str, Any]]]]
) -> Dict[str, 'CreditAssignmentOp']:
    """
    Creates a dictionary of credit assignment ops from class names and args.

    Args:
        seg_len (int): Trajectory segment length for credit assignment.
        extra_steps (int): Extra steps for n-step return-based credit assignment.
            Should equal n-1 when n steps are used.
        credit_assignment_spec (Mapping[str, Mapping[str, Union[str, Mapping[str, Any]]]]):
            Mapping of reward names to a dictionary with keys 'cls_name' and
            'cls_args'. The 'cls_name' key should map to the name of a derived
            class of `CreditAssignmentOp`. The 'cls_args' key should map to
            constructor's reward-varying arguments (i.e., the arguments besides
            seg_len and extra_steps).

    Returns:
        Dict[str, CreditAssignmentOp]: Dictionary of credit assignment ops,
        keyed by reward name.
    """
    ops = dict()
    for reward_name in credit_assignment_spec:
        op_spec = credit_assignment_spec[reward_name]
        op = get_credit_assignment_op(
            cls_name=op_spec['cls_name'],
            cls_args={
                'seg_len': seg_len,
                'extra_steps': extra_steps,
                **op_spec['cls_args']
            }
        )
        ops[reward_name] = op
    return ops


class CreditAssignmentOp(metaclass=abc.ABCMeta):
    """
    Abstract class for credit assignment operations.
    """
    def __init__(
            self,
            seg_len: int,
            extra_steps: int,
            gamma: float,
            use_dones: bool
    ):
        """
        Args:
            seg_len (int): Trajectory segment length for credit assignment.
            extra_steps (int): Extra steps for n-step return-based credit
                assignment. Should equal n-1 when n steps are used.
            gamma (float): Discount factor in [0, 1).
            use_dones (bool): Whether or not to block credit assignment across
                episodes. Should be True for conventional RL settings.
                Intended to be False for use with certain intrinsic rewards
                like RND (Burda et al., 2018), or with certain meta-RL
                algorithms like RL^2 (Duan et al., 2016).
        """
        self._seg_len = seg_len
        self._extra_steps = extra_steps
        self._gamma = gamma
        self._use_dones = use_dones


class AdvantageEstimator(CreditAssignmentOp, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def estimate_advantages(self, rewards, vpreds, dones):
        """
        Estimate advantages.

        Args:
            rewards (torch.Tensor): Torch tensor of rewards at each timestep,
                with shape [seg_len + extra_steps].
            vpreds (torch.Tensor): Torch tensor of value predictions at each
                timestep, with shape [seg_len + extra_steps + 1].
            dones (torch.Tensor): Torch tensor of done signals at each timestep,
                with shape [seg_len + extra_steps].

        Returns:
            torch.Tensor: Torch tensor of advantage estimates with shape [seg_len].
        """


class BellmanOperator(CreditAssignmentOp, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def estimate_action_values(self, rewards, qpreds, dones, **kwargs):
        """
        Estimate action values.

        Args:
            rewards (torch.Tensor): Torch tensor of rewards at each timestep,
                with shape [seg_len + extra_steps].
            qpreds (torch.Tensor): Torch tensor of action-value predictions
                at each timestep, with shape [seg_len + extra_steps + 1].
            dones (torch.Tensor): Torch tensor of done signals at each timestep,
                with shape [seg_len + extra_steps].

        Returns:
            torch.Tensor: Torch tensor of action-value estimates with shape [seg_len].
        """


class GAE(AdvantageEstimator):
    """
    Generalized Advantage Estimation.

    Reference:
        Schulman et al., 2016 -
            'High Dimensional Continuous Control with Generalized Advantage Estimation'
    """
    def __init__(self, seg_len, extra_steps, gamma, use_dones, lambda_):
        super().__init__(seg_len, extra_steps, gamma, use_dones)
        self._lambda = lambda_

    def estimate_advantages(self, rewards, vpreds, dones):
        advantages = tc.zeros(self._seg_len+self._extra_steps+1, dtype=tc.float32)
        for t in reversed(range(0, self._seg_len + self._extra_steps)):  # T+(n-1)-1, ..., 0
            r_t = rewards[t]
            V_t = vpreds[t]
            V_tp1 = vpreds[t+1]
            A_tp1 = advantages[t+1]
            nonterminal_t = (1.-dones[t]) if self._use_dones else 1.
            delta_t = -V_t + r_t + nonterminal_t * self._gamma * V_tp1
            A_t = delta_t + nonterminal_t * self._gamma * self._lambda * A_tp1
            advantages[t] = A_t
        return advantages


class NStepAdvantageEstimator(AdvantageEstimator):
    """
    N-step advantage estimator.

    Reference:
        V. Mnih et al., 2016 -
            'Asynchronous Methods for Deep Reinforcement Learning'.
    """
    def __init__(self, seg_len, extra_steps, gamma, use_dones):
        super().__init__(seg_len, extra_steps, gamma, use_dones)

    def estimate_advantages(self, rewards, vpreds, dones):
        # r_t + gamma * r_tp1 + ... + gamma^nm1 * r_tpnm1 + gamma^n * V(s_tpn)
        # todo: think about this more and add a unit test.
        #  extra steps equals n-1 for n-step returns.
        advantages = tc.zeros(self._seg_len, dtype=tc.float32)
        for t in reversed(range(0, self._seg_len)):  # T-1, ..., 0
            V_tpn = vpreds[t+self._extra_steps+1]    # V[t+(n-1)+1] = V[t+n]
            R_t = V_tpn
            for s in reversed(range(0, self._extra_steps+1)):  # ((n-1)+1)-1 = n-1, ..., 0
                r_tps = rewards[t+s]                           # r[t+n-1], ..., r[t+0].
                nonterminal_tps = (1.-dones[t+s]) if self._use_dones else 1.
                R_t = r_tps + nonterminal_tps * self._gamma * R_t
            V_t = vpreds[t]
            advantages[t] = R_t - V_t
        return advantages


class SimpleDiscreteBellmanOptimalityOperator(BellmanOperator):
    """
    Simple (non-distributional) discrete-action Bellman optimality operator.

    Reference:
        V. Mnih et al., 2015 -
            'Human Level Control through Deep Reinforcement Learning'
        H. van Hasselt et al., 2015 -
            'Deep Reinforcement Learning with Double Q-learning'
    """
    def __init__(self, seg_len, extra_steps, gamma, use_dones, double_q):
        """
        Args:
            seg_len (int): Trajectory segment length for credit assignment.
            extra_steps (int): Extra steps for n-step return-based credit assignment.
                Should equal n-1 when n steps are used.
            gamma (float): Discount factor in [0, 1).
            use_dones (bool): Whether or not to block credit assignment across
                episodes. Should be True for conventional RL settings.
                Intended to be False for use with certain intrinsic rewards
                like RND (Burda et al., 2018), or with certain meta-RL
                algorithms like RL^2 (Duan et al., 2016).
            double_q (bool): Use double-Q learning?
        """
        super().__init__(seg_len, extra_steps, gamma, use_dones)
        self._double_q = double_q

    def estimate_action_values(self, rewards, qpreds, tgt_qpreds, dones):
        # r_t + gamma * r_tp1 + ... + gamma^nm1 * r_tpnm1 + gamma^n * Q(s_tpn, a_tpn)
        # todo: think about this more and add a unit test.
        #  extra steps equals n-1 for n-step returns.
        bellman_backups = tc.zeros(self._seg_len, dtype=tc.float32)
        for t in reversed(range(0, self._seg_len)):  # T-1, ..., 0
            Qs_tpn_tgt = tgt_qpreds[t+self._extra_steps+1]  # Q[t+(n-1)+1] = Q[t+n]
            if self._double_q:
                Qs_tpn = qpreds[t+self._extra_steps+1]      # Q[t+(n-1)+1] = Q[t+n]
                greedy_a_tpn = tc.argmax(Qs_tpn, dim=-1)
            else:
                greedy_a_tpn = tc.argmax(Qs_tpn_tgt, dim=-1)
            Q_tpn_tgt = tc.gather(
                input=Qs_tpn_tgt, dim=-1,
                index=greedy_a_tpn.unsqueeze(-1)
            ).squeeze(-1)
            R_t = Q_tpn_tgt
            for s in reversed(range(0, self._extra_steps+1)):  # ((n-1)+1)-1 = n-1, ..., 0
                r_tps = rewards[t+s]                           # r[t+n-1], ..., r[t+0].
                nonterminal_tps = (1.-dones[t+s]) if self._use_dones else 1.
                R_t = r_tps + nonterminal_tps * self._gamma * R_t
            bellman_backups[t] = R_t
        return bellman_backups
