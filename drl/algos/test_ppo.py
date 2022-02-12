import torch as tc

from drl.algos.ppo import (
    ppo_policy_entropy_bonus, ppo_policy_surrogate_objective, ppo_vf_loss, PPO)


def test_ppo_policy_entropy_bonus():
    entropy_at_visited_states = tc.tensor([0.0, 1.0, 2.0, 3.0])
    entropy_dict = ppo_policy_entropy_bonus(
        entropies=entropy_at_visited_states, ent_coef=0.01)
    tc.testing.assert_close(
        actual=entropy_dict['state_averaged_entropy'], expected=tc.tensor(1.5))
    tc.testing.assert_close(
        actual=entropy_dict['policy_entropy_bonus'], expected=tc.tensor(0.015))


def test_ppo_policy_surrogate_objective():
    logprobs_new = tc.log(tc.tensor([0.105, 0.08, 0.24, 0.095, 0.12, 0.16]))
    logprobs_old = tc.log(tc.tensor([0.10, 0.10, 0.20, 0.10, 0.10, 0.20]))
    ratios_expected = tc.tensor([1.05, 0.80, 1.20, 0.95, 1.20, 0.80])
    tc.testing.assert_close(
        actual=tc.exp(logprobs_new - logprobs_old), expected=ratios_expected)

    advantages = tc.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    # clipped = tc.tensor([False, False, True, False, False, True])
    expected = tc.tensor([
        1.0 * 1.05,
        1.0 * 0.80,
        1.0 * 1.10,
        -1.0 * 0.95,
        -1.0 * 1.20,
        -1.0 * 0.90
    ]).mean()
    policy_dict = ppo_policy_surrogate_objective(
        logprobs_new=logprobs_new,
        logprobs_old=logprobs_old,
        advantages={'extrinsic': advantages},
        clip_param=0.10,
        reward_weights={'extrinsic': 1.0})
    actual = policy_dict['policy_surrogate_objective']
    tc.testing.assert_close(actual=actual, expected=expected)


def test_ppo_vf_loss():
    vpreds_new = tc.tensor([1.05, 2.4, 1.2, -1.05, -2.4, -1.2])
    vpreds_old = tc.tensor([1.0, 2.0, 1.0, -1.0, -2.0, -1.0])
    td_lam_rets = tc.tensor([1.5, 1.5, 1.5, -1.5, -1.5, -1.5])
    #clipped = tc.tensor([False, False, True, False, False, True])
    expected_noclip = tc.tensor([
        (1.05 - 1.5)**2,
        (2.4 - 1.5)**2,
        (1.2 - 1.5)**2,
        (-1.05 - -1.5)**2,
        (-2.4 - -1.5)**2,
        (-1.2 - -1.5)**2,
    ]).mean()
    expected_clip = tc.tensor([
        (1.05 - 1.5)**2,
        (2.4 - 1.5)**2,
        (1.1 - 1.5)**2,
        (-1.05 - -1.5)**2,
        (-2.4 - -1.5)**2,
        (-1.1 - -1.5)**2,
    ]).mean()

    value_dict_noclip = ppo_vf_loss(
        vpreds_new={'extrinsic': vpreds_new},
        vpreds_old={'extrinsic': vpreds_old},
        td_lambda_returns={'extrinsic': td_lam_rets},
        clip_param=0.10,
        vf_loss_criterion=tc.nn.MSELoss(reduction='none'),
        vf_loss_clipping=False,
        vf_simple_weighting=True,
        reward_weights={'extrinsic': 1.0})
    actual_noclip = value_dict_noclip['vf_loss']
    tc.testing.assert_close(actual=actual_noclip, expected=expected_noclip)

    value_dict_clip = ppo_vf_loss(
        vpreds_new={'extrinsic': vpreds_new},
        vpreds_old={'extrinsic': vpreds_old},
        td_lambda_returns={'extrinsic': td_lam_rets},
        clip_param=0.10,
        vf_loss_criterion=tc.nn.MSELoss(reduction='none'),
        vf_loss_clipping=True,
        vf_simple_weighting=True,
        reward_weights={'extrinsic': 1.0})
    actual_clip = value_dict_clip['vf_loss']
    tc.testing.assert_close(actual=actual_clip, expected=expected_clip)
