from collections import Counter

import torch as tc


def global_mean(metric, world_size):
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.item() / world_size


def global_means(metrics, world_size):
    # for logging purposes only!
    return Counter({k: global_mean(v, world_size) for k, v in metrics.items()})


def trajectory_segment_generator(env, nets, segment_length):
    t = 0
    o_t = env.reset()

    observations = list()
    actions = list()
    rewards = dict()
    dones = list()

    ep_lens, curr_ep_len = list(), 0
    ep_rets, curr_ep_ret = list(), 0.
    ep_lens_raw, curr_ep_len_raw = list(), 0
    ep_rets_raw, curr_ep_ret_raw = list(), 0.

    while True:
        if t % segment_length == 0:
            if t > 0:
                observations.append(o_t)
                yield {
                    "observations": tc.tensor(observations).clone(),
                    "actions": tc.tensor(actions).clone(),
                    "rewards": tc.tensor(rewards).clone(),
                    "dones": tc.tensor(dones).clone(),
                    "ep_lens": ep_lens,
                    "ep_rets": ep_rets,
                    "ep_lens_raw": ep_lens_raw,
                    "ep_rets_raw": ep_rets_raw
                }
                observations = list()
                actions = list()
                rewards = list()
                dones = list()
                ep_lens = list()
                ep_rets = list()
                ep_lens_raw = list()
                ep_rets_raw = list()

        for net in nets.values():
            net.eval()
        if 'policy_net' in nets:
            policy_net = nets.get('policy_net')
        else:
            policy_net = nets.get('q_network')
        predictions = policy_net(
            x=tc.FloatTensor(o_t).unsqueeze(0), predictions=['policy'])
        pi_dist_t = predictions.get('policy')
        a_t = pi_dist_t.sample()
        o_tp1, r_t, done_t, info_t = env.step(a_t.squeeze(0).detach().numpy())
        if not isinstance(r_t, dict):
            r_t = dict({'extrinsic_raw': r_t, 'extrinsic': r_t})
        r_t_raw, r_t = r_t['extrinsic_raw'], r_t['extrinsic']

        observations.append(tc.tensor(o_t).float())
        actions.append(tc.tensor(a_t).long())
        for key in r_t:
            if key != 'extrinsic_raw':
                rewards[key].append(tc.tensor(r_t[key]).float())
        dones.append(tc.tensor(done_t).int())

        curr_ep_len += 1
        curr_ep_ret += r_t['extrinsic']
        curr_ep_ret_raw += r_t['extrinsic_raw']

        if done_t:
            ep_lens.append(curr_ep_len)
            ep_rets.append(curr_ep_ret)
            curr_ep_len = 0
            curr_ep_ret = 0.

            def was_real_done():
                if 'ale.lives' in info_t:
                    return info_t['ale.lives'] == 0
                return True
            if was_real_done():
                ep_lens_raw.append(curr_ep_len_raw)
                ep_rets_raw.append(curr_ep_ret_raw)
                curr_ep_len_raw = 0
                curr_ep_ret_raw = 0.
            o_tp1 = env.reset()
            # note: episodic life wrapper blocks true reset if not true done

        t += 1
        o_t = o_tp1
