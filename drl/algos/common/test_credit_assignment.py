import torch as tc

from drl.algos.common.credit_assignment import (
    GAE, NStepAdvantageEstimator, SimpleDiscreteBellmanOptimalityOperator)


def test_gae():
    seg_len = 2
    extra_steps = 0
    gamma = 0.99
    lambda_ = 0.95
    rewards = tc.tensor([0., 0.])
    value_estimates = tc.tensor([0., 0., 1.])

    dones_1 = tc.tensor([0., 0.])
    credit_assignment_op_1 = GAE(
        seg_len=seg_len,
        extra_steps=extra_steps,
        gamma=gamma,
        use_dones=True,
        lambda_=lambda_)
    advantages = credit_assignment_op_1.estimate_advantages(
        rewards=rewards, vpreds=value_estimates, dones=dones_1)
    advantages_expected = tc.tensor([
        rewards[0] + gamma * value_estimates[1] + lambda_ *
        (rewards[0] + gamma * rewards[1] + (gamma**2) * value_estimates[2]),
        rewards[1] + gamma * value_estimates[2]
    ])
    tc.testing.assert_close(actual=advantages, expected=advantages_expected)

    dones_2 = tc.tensor([1., 0.])
    credit_assignment_op_2 = credit_assignment_op_1
    advantages = credit_assignment_op_2.estimate_advantages(
        rewards=rewards, vpreds=value_estimates, dones=dones_2)
    advantages_expected = tc.tensor([
        rewards[0] + (1 - dones_2[0]) * gamma * value_estimates[1] + lambda_ * (
            rewards[0] + (1 - dones_2[0]) * gamma * rewards[1] +
            (1 - dones_2[0]) * (1 - dones_2[1]) *
            (gamma**2) * value_estimates[2]),
        rewards[1] + (1 - dones_2[1]) * gamma * value_estimates[2]
    ])
    tc.testing.assert_close(actual=advantages, expected=advantages_expected)

    dones_3 = tc.tensor([1., 0.])
    credit_assignment_op_3 = GAE(
        seg_len=seg_len,
        extra_steps=extra_steps,
        gamma=gamma,
        use_dones=False,
        lambda_=lambda_)
    advantages = credit_assignment_op_3.estimate_advantages(
        rewards=rewards, vpreds=value_estimates, dones=dones_3)
    advantages_expected = tc.tensor([
        rewards[0] + 1 * gamma * value_estimates[1] + lambda_ * (
            rewards[0] + 1 * gamma * rewards[1] + 1 * 1 *
            (gamma**2) * value_estimates[2]),
        rewards[1] + 1 * gamma * value_estimates[2]
    ])
    tc.testing.assert_close(actual=advantages, expected=advantages_expected)


def test_nstep_advantages():
    seg_len = 2
    extra_steps = 2
    gamma = 0.99
    rewards = tc.tensor([1., 2., 3., 4.])
    value_estimates = tc.tensor([0., 0., 0., 0., 5.])

    dones_1 = tc.tensor([0., 0., 0., 0., 0.])
    credit_assignment_op_1 = NStepAdvantageEstimator(
        seg_len=seg_len, extra_steps=extra_steps, gamma=gamma, use_dones=True)
    advantages_1 = credit_assignment_op_1.estimate_advantages(
        rewards=rewards, vpreds=value_estimates, dones=dones_1)
    advantages_expected_1 = tc.tensor([
        rewards[0] + gamma * rewards[1] + (gamma**2) * rewards[2] +
        (gamma**3) * value_estimates[3],
        rewards[1] + gamma * rewards[2] + (gamma**2) * rewards[3] +
        (gamma**3) * value_estimates[4],
    ])
    tc.testing.assert_close(actual=advantages_1, expected=advantages_expected_1)

    dones_2 = tc.tensor([0., 1., 0., 0.])
    credit_assignment_op_2 = credit_assignment_op_1
    advantages_2 = credit_assignment_op_2.estimate_advantages(
        rewards=rewards, vpreds=value_estimates, dones=dones_2)
    advantages_expected_2 = tc.tensor([
        rewards[0] + (1 - dones_2[0]) * gamma * rewards[1] + (1 - dones_2[0]) *
        (1 - dones_2[1]) * (gamma**2) * rewards[2] + (1 - dones_2[0]) *
        (1 - dones_2[1]) * (1 - dones_2[2]) * (gamma**3) * value_estimates[3],
        rewards[1] + (1 - dones_2[1]) * gamma * rewards[2] + (1 - dones_2[1]) *
        (1 - dones_2[2]) * (gamma**2) * rewards[3] + (1 - dones_2[1]) *
        (1 - dones_2[2]) * (1 - dones_2[3]) * (gamma**3) * value_estimates[4]
    ])
    tc.testing.assert_close(actual=advantages_2, expected=advantages_expected_2)

    dones_3 = dones_2
    credit_assignment_op_3 = NStepAdvantageEstimator(
        seg_len=seg_len, extra_steps=extra_steps, gamma=gamma, use_dones=False)
    advantages_3 = credit_assignment_op_3.estimate_advantages(
        rewards=rewards, vpreds=value_estimates, dones=dones_3)
    tc.testing.assert_close(actual=advantages_3, expected=advantages_expected_1)


def test_simple_discrete_bellman_optimality_op():
    num_actions = 3
    seg_len = 2
    extra_steps = 0
    gamma = 0.99
    rewards = tc.tensor([0., 0.])
    qpreds = tc.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 1.]])
    tgt_qpreds = tc.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    dones = tc.tensor([0., 0.])

    credit_assignment_op_1 = SimpleDiscreteBellmanOptimalityOperator(
        seg_len=seg_len,
        extra_steps=extra_steps,
        gamma=gamma,
        use_dones=True,
        double_q=False)
    operator_image_actual = credit_assignment_op_1.estimate_action_values(
        rewards=rewards, qpreds=qpreds, tgt_qpreds=tgt_qpreds, dones=dones)
    operator_image_expected = tc.tensor([
        rewards[0] + gamma * tgt_qpreds[1, 0],
        rewards[1] + gamma * tgt_qpreds[2, 1]
    ])
    tc.testing.assert_close(
        actual=operator_image_actual, expected=operator_image_expected)

    credit_assignment_op_2 = SimpleDiscreteBellmanOptimalityOperator(
        seg_len=seg_len,
        extra_steps=extra_steps,
        gamma=gamma,
        use_dones=True,
        double_q=True)
    operator_image_actual = credit_assignment_op_2.estimate_action_values(
        rewards=rewards, qpreds=qpreds, tgt_qpreds=tgt_qpreds, dones=dones)
    operator_image_expected = tc.tensor([
        rewards[0] + gamma * tgt_qpreds[1, 2],
        rewards[1] + gamma * tgt_qpreds[2, 2]
    ])
    tc.testing.assert_close(
        actual=operator_image_actual, expected=operator_image_expected)

    dones_3 = tc.tensor([1., 0.])
    credit_assignment_op_3 = SimpleDiscreteBellmanOptimalityOperator(
        seg_len=seg_len,
        extra_steps=extra_steps,
        gamma=gamma,
        use_dones=False,
        double_q=True)
    operator_image_actual = credit_assignment_op_3.estimate_action_values(
        rewards=rewards, qpreds=qpreds, tgt_qpreds=tgt_qpreds, dones=dones_3)
    operator_image_expected = tc.tensor([
        rewards[0] + (1 - dones_3[0]) * gamma * tgt_qpreds[1, 2],
        rewards[1] + (1 - dones_3[1]) * gamma * tgt_qpreds[2, 2]
    ])
    tc.testing.assert_close(
        actual=operator_image_actual, expected=operator_image_expected)
