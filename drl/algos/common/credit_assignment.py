import torch as tc


def extract_reward_name(predictor_name):
    prefixes = ['value_', 'action_value_', 'policy_']
    for prefix in prefixes:
        if predictor_name.startswith(prefix):
            return predictor_name[len(prefix):]
    raise ValueError("Unrecognized predictor name.")


def gae_advantages(seg_len, extra_steps, gamma, lam, rewards, vpreds, dones):
    advantages = tc.zeros(seg_len+extra_steps+1, dtype=tc.float32)
    for t in reversed(range(0, seg_len+extra_steps)):  # T+(n-1)-1, ..., 0
        r_t = rewards[t]
        V_t = vpreds[t]
        V_tp1 = vpreds[t+1]
        A_tp1 = advantages[t+1]
        delta_t = -V_t + r_t + (1. - dones[t]) * gamma * V_tp1
        A_t = delta_t + (1. - dones[t]) * gamma * lam * A_tp1
        advantages[t] = A_t
    return advantages


def nstep_advantages(seg_len, extra_steps, gamma, rewards, vpreds, dones):
    # r_t + gamma * r_tp1 + ... + gamma^nm1 * r_tpnm1 + gamma^n * V(s_tpn)
    # todo: think about this more and add a unit test.
    #  extra steps equals n-1 for n-step returns.
    advantages = tc.zeros(seg_len, dtype=tc.float32)
    for t in reversed(range(0, seg_len)):  # T-1, ..., 0
        V_tpn = vpreds[t+extra_steps+1]    # V[t+(n-1)+1] = V[t+n]
        R_t = V_tpn
        for s in reversed(range(0, extra_steps+1)):  # ((n-1)+1)-1 = n-1, ..., 0
            r_tps = rewards[t+s]                     # r[t+n-1], ..., r[t+0].
            R_t = r_tps + (1. - dones[t+s]) * gamma * R_t
        V_t = vpreds[t]
        advantages[t] = R_t - V_t
    return advantages
