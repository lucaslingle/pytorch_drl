import abc
import importlib

import torch as tc


def extract_reward_name(predictor_name):
    prefixes = ['value_', 'action_value_']
    for prefix in prefixes:
        if predictor_name.startswith(prefix):
            return predictor_name[len(prefix):]
    raise ValueError("Unrecognized predictor name.")


def get_credit_assignment_op(cls_name, cls_args):
    module = importlib.import_module('drl.algos.common.credit_assignment')
    cls = getattr(module, cls_name)
    return cls(**cls_args)


def get_credit_assignment_ops(seg_len, extra_steps, credit_assignment_spec):
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
    def __init__(self, seg_len, extra_steps, gamma, use_dones=True):
        """
        Args:
            seg_len: Trajectory segment length for credit assignment.
            extra_steps: Extra steps for n-step return-based credit assignment.
                Should equal n-1 when n steps are used.
            gamma: Discount factor in [0, 1).
            use_dones: Whether or not to block credit assignment across episodes.
                Intended for use with intrinsic rewards or algorithms like RL^2
                (Duan et al., 2016).
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
            rewards: Torch tensor of rewards at each timestep,
                with shape [seg_len + extra_steps].
            vpreds: Torch tensor of value predictions at each timestep,
                with shape [seg_len + extra_steps + 1].
            dones: Torch tensor of done signals at each timestep,
                with shape [seg_len + extra_steps].

        Returns:
            Torch tensor of advantage estimates with shape [seg_len].
        """


class GAE(AdvantageEstimator):
    """
    Generalized Advantage Estimation (Schulman et al., 2016).
    """
    def __init__(self, seg_len, extra_steps, gamma, use_dones, lambda_):
        super().__init__(seg_len, extra_steps, gamma, use_dones)
        self._lambda = lambda_

    def estimate_advantages(self, rewards, vpreds, dones):
        advantages = tc.zeros(self._seg_len+self._extra_steps+1, dtype=tc.float32)
        for t in reversed(range(0, self._seg_len + self._extra_steps)):  # T+(n-1)-1, ..., 0
            r_t = rewards[t]
            V_t = vpreds[t]
            V_tp1 = vpreds[t + 1]
            A_tp1 = advantages[t + 1]
            nonterminal_t = (1. - dones[t]) if self._use_dones else 1.
            delta_t = -V_t + r_t + nonterminal_t * self._gamma * V_tp1
            A_t = delta_t + nonterminal_t * self._gamma * self._lambda * A_tp1
            advantages[t] = A_t
        return advantages


class NStepAdvantageEstimator(AdvantageEstimator):
    """
    N-step advantage estimation (Mnih et al., 2016).
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
            for s in reversed(range(0, self._extra_steps + 1)):  # ((n-1)+1)-1 = n-1, ..., 0
                r_tps = rewards[t+s]                             # r[t+n-1], ..., r[t+0].
                nonterminal_tps = (1.-dones[t+s]) if self._use_dones else 1.
                R_t = r_tps + nonterminal_tps * self._gamma * R_t
            V_t = vpreds[t]
            advantages[t] = R_t - V_t
        return advantages
