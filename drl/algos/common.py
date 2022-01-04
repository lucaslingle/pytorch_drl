from collections import Counter
import copy

import torch as tc

from drl.envs.wrappers import Wrapper


def global_mean(metric, world_size):
    global_metric = metric.clone().float().detach()
    tc.distributed.all_reduce(global_metric, op=tc.distributed.ReduceOp.SUM)
    return global_metric.item() / world_size


def global_means(metrics, world_size):
    return Counter({k: global_mean(v, world_size) for k, v in metrics.items()})


class MetadataManager:
    def __init__(self, present_meta):
        self._fields = present_meta.keys()
        self._present_defaults = copy.deepcopy(present_meta)
        self._present_meta = present_meta
        self._past_meta = {key: list() for key in self._fields}

    def update_present(self, deltas):
        for key in deltas:
            self._present_meta[key] += deltas[key]

    def present_done(self, fields):
        for key in fields:
            self._past_meta[key].append(self._present_meta[key])
            self._present_meta[key] = copy.deepcopy(self._present_defaults[key])

    @property
    def present_meta(self):
        return self._present_meta

    @property
    def past_meta(self):
        return self._past_meta


class Trajectory:
    def __init__(self, obs_shape, rew_keys, seg_len):
        self._rew_keys = rew_keys
        self._seg_len = seg_len
        self._observations = tc.zeros((seg_len+1, *obs_shape), dtype=tc.float32)
        self._actions = tc.zeros(seg_len, dtype=tc.float32)
        self._rewards = {
            k: tc.zeros(seg_len, dtype=tc.float32) for k in rew_keys
        }
        self._dones = tc.zeros(rew_keys, dtype=tc.float32)

    def record(self, t, o_t, a_t, r_t, d_t):
        i = t % self._seg_len
        self._observations[i] = tc.tensor(o_t).float()
        self._actions[i] = tc.tensor(a_t).long()
        for key in self._rew_keys:
            self._rewards[key][i] = tc.tensor(r_t[key]).float()
        self._dones[i] = tc.tensor(d_t).float()

    def report(self):
        return {
            'observations': self._observations,
            'actions': self._actions,
            'rewards': self._rewards,
            'dones': self._dones
        }


class TrajectoryManager:
    def __init__(self, env, policy_net, segment_length):
        self._env = env
        self._policy_net = policy_net
        self._segment_length = segment_length
        self._o_t = env.reset()
        self._metadata_mgr = MetadataManager(
            present_meta={
                'ep_len': 0,
                'ep_ret': 0.,
                'ep_len_raw': 0,
                'ep_ret_raw': 0.
            }
        )

    def _get_reward_keys(self):
        def spec_exists():
            if isinstance(self._env, Wrapper):
                return self._env.reward_spec is not None
            return False
        if spec_exists():
            return self._env.reward_spec.keys
        return {'extrinsic_raw', 'extrinsic'}

    def generate(self):
        """
        Generate trajectory experience and metadata.
        """
        # instantiate a trajectory instance
        trajectory = Trajectory(
            obs_shape=self._o_t.shape,
            rew_keys=self._get_reward_keys(),
            seg_len=self._segment_length)

        # turn off dropout, batchnorm, etc.
        # these should only be used to *train* off-policy algorithms, if at all.
        # note if we implement noisy nets it will be in a way that is unaffected by this.
        self._policy_net.eval()

        # generate a trajectory segment
        for t in range(0, self._segment_length):
            # choose action
            predictions = self._policy_net(
                tc.tensor(self._o_t).float().unsqueeze(0), predictions=['policy'])  # todo(lucaslingle): edit EpsilonGreedyCategoricalPolicy to require an inputted schedule during creation
            pi_dist_t = predictions.get('policy')
            a_t = pi_dist_t.sample().squeeze(0).detach().numpy()

            # step environment
            o_tp1, r_t, done_t, info_t = self._env.step(a_t)
            if not isinstance(r_t, dict):
                r_t = {'extrinsic_raw': r_t, 'extrinsic': r_t}

            # record everything
            trajectory.record(t, self._o_t, a_t, r_t, done_t)
            self._metadata_mgr.update_present(
                deltas={
                    'ep_len': 1,
                    'ep_ret': r_t['extrinsic'],
                    'ep_len_raw': 1,
                    'ep_ret_raw': r_t['extrinsic_raw']
                }
            )

            # handle metadata for episode boundaries
            if done_t:
                self._metadata_mgr.present_done(fields=['ep_len', 'ep_ret'])
                def was_real_done():
                    if 'ale.lives' in info_t:
                        return info_t['ale.lives'] == 0
                    return True
                if was_real_done():
                    self._metadata_mgr.present_done(
                        fields=['ep_len_raw', 'ep_ret_raw'])
                o_tp1 = self._env.reset()

            # save next observation
            self._o_t = o_tp1

        # return results with next timestep observation included
        results = {
            **trajectory.report(),
            'metadata': self._metadata_mgr.past_meta
        }
        results['observations'][-1] = o_tp1
        return results
