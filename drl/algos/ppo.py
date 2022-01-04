import torch as tc
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from drl.algos.abstract import Algo
from drl.algos.common import TrajectoryManager, global_means


class PPO(Algo):
    def __init__(self, rank, config):
        super().__init__(rank, config)
        self._learning_system = self._get_learning_system()
        self._trajectory_mgr = TrajectoryManager(
            env=self._learning_system['env'],
            policy_net=self._learning_system['policy_net'],
            segment_length=self._config['algo']['segment_length'])
        if self._rank == 0:
            self._writer = SummaryWriter(self._config.get('log_dir'))

    def _get_learning_system(self):
        env_config = self._config.get('env')
        env = self._get_env(env_config)

        policy_config = self._config.get('policy_net')
        policy_net = self._get_net(policy_config)
        policy_optimizer_config = policy_config.get('optimizer')
        policy_optimizer = self._get_opt(policy_optimizer_config, policy_net)

        value_config = self._config.get('value_net')
        value_net, value_optimizer = None, None
        if not value_config.get('use_shared_architecture'):
            value_net = self._get_net(value_config)
            value_optimizer_config = value_config.get('optimizer')
            value_optimizer = self._get_opt(value_optimizer_config, value_net)

        checkpointables = {
            'policy_net': policy_net,
            'policy_optimizer': policy_optimizer,
            'value_net': value_net,
            'value_optimizer': value_optimizer
        }
        checkpointables_ = {k: v for k,v in checkpointables.items()}
        checkpointables_.update(env.get_checkpointables())
        global_step = self._maybe_load_checkpoints(checkpointables_, step=None)
        return {'global_step': global_step, 'env': env, **checkpointables}

    @tc.no_grad()
    def _annotate(self, trajectory, policy_net, value_net):
        algo_config = self._config.get('algo')

        # get trajectory variables.
        observations = trajectory.get('observations')
        actions = trajectory.get('actions')
        rewards = trajectory.get('rewards')
        dones = trajectory.get('dones')

        # decide who should predict what.
        reward_keys = rewards.keys()
        relevant_reward_keys = [k for k in reward_keys if k != 'extrinsic_raw']
        policy_predict = ['policy']
        value_predict = []
        for key in relevant_reward_keys:
            value_predict.append(f'value_{key}')
        if value_net is None:
            policy_predict.extend(value_predict)

        # compute logprobs of actions.
        predictions = policy_net(observations, policy_predict)
        logprobs = predictions.get('policy').log_prob(actions)
        logprobs = logprobs[0:-1]

        # compute value estimates.
        if value_net is None:
            vpreds = {k: predictions[k] for k in value_predict}
        else:
            vpreds = value_net(observations, value_predict)
        vpreds = {k.partition('_')[2]: vpreds[k] for k in vpreds}

        # compute GAE(lambda), TD(lambda) estimators.
        seg_len = algo_config.get('segment_length')
        lam = algo_config.get('gae_lambda')
        gamma = algo_config.get('discount_gamma')
        advantages = {
            k: tc.zeros(seg_len+1, dtype=tc.float32)
            for k in relevant_reward_keys
        }
        for k in relevant_reward_keys:
            for t in reversed(range(0, seg_len)):  # T-1, ..., 0
                r_t = rewards[k][t]
                V_t = vpreds[k][t]
                V_tp1 = vpreds[k][t+1]
                A_tp1 = advantages[k][t+1]
                delta_t = -V_t + r_t + (1. - dones[t]) * gamma[k] * V_tp1
                A_t = delta_t + (1. - dones[t]) * gamma[k] * lam[k] * A_tp1
                advantages[k][t] = A_t
        advantages = {k: advantages[k][0:-1] for k in advantages}
        vpreds = {k: vpreds[k][0:-1] for k in vpreds}
        td_lam_rets = {k: advantages[k] + vpreds[k] for k in advantages}
        observations = observations[0:-1]

        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'logprobs': logprobs,
            'vpreds': vpreds,
            'advantages': advantages,
            'td_lambda_returns': td_lam_rets
        }

    def _slice_minibatch(self, trajectory, indices):
        results = dict()
        for field in trajectory:
            if isinstance(trajectory[field], dict):
                slice = {k: v[indices] for k,v in trajectory[field].items()}
            else:
                slice = trajectory[field][indices]
            results[field] = slice
        return results

    def _compute_losses(self, mb, policy_net, value_net, clip_param, ent_coef):
        # decide who should predict what.
        reward_keys = mb['rewards'].keys()
        relevant_reward_keys = [k for k in reward_keys if k != 'extrinsic_raw']
        policy_predict = ['policy']
        value_predict = [f'value_{k}' for k in relevant_reward_keys]
        if value_net is None:
            policy_predict.extend(value_predict)

        # compute logprobs of actions.
        predictions_new = policy_net(mb['observations'], policy_predict)
        pi_new = predictions_new.get('policy')
        logprobs_new = pi_new.log_prob(mb['actions'])

        # compute value estimates.
        if value_net is None:
            vpreds_new = {k: predictions_new[k] for k in value_predict}
        else:
            vpreds_new = value_net(mb['observations'], value_predict)
        vpreds_new = {k.partition('_')[2]: vpreds_new[k] for k in vpreds_new}

        # entropy
        entropies = pi_new.entropy()
        mean_entropy = tc.mean(entropies)
        policy_entropy_bonus = ent_coef * mean_entropy

        # ppo policy loss, value loss
        policy_ratio = tc.exp(logprobs_new - mb['logprobs'])
        clipped_policy_ratio = tc.clip(policy_ratio, 1-clip_param, 1+clip_param)
        policy_surrogate_objective = 0.
        vf_loss = 0.
        clipfrac = 0.
        for reward_id, key in enumerate(relevant_reward_keys):
            surr1 = mb['advantages'][key] * policy_ratio
            surr2 = mb['advantages'][key] * clipped_policy_ratio
            ppo_surr_for_reward = tc.mean(tc.min(surr1, surr2))
            vf_loss_for_reward = tc.mean(tc.square(mb['td_lambda'][key] - vpreds_new[key]))
            clipfrac_for_reward = tc.mean(tc.greater(surr1, surr2).float())
            if len(relevant_reward_keys) == 1:
                policy_surrogate_objective += ppo_surr_for_reward
                vf_loss += vf_loss_for_reward
            else:
                weight = self._config['algo']['reward_weightings'][key]
                policy_surrogate_objective += weight * ppo_surr_for_reward  # this weighting is equiv to using weight on the rewards
                vf_loss += tc.square(weight) * vf_loss_for_reward  # todo(lucaslingle): investigate if official implementation of rnd uses this weighting
            clipfrac += clipfrac_for_reward

        policy_loss = -(policy_surrogate_objective + policy_entropy_bonus)
        if value_net is None:
            weight = self._config['algo']['vf_loss_weight']
        else:
            weight = 1.
        composite_loss = policy_loss + weight * vf_loss
        return {
            'policy_loss': policy_loss,
            'value_loss': vf_loss,
            'composite_loss': composite_loss,
            'meanent': mean_entropy,
            'clipfrac': clipfrac / len(relevant_reward_keys)
        }

    def training_loop(self):
        policy_net = self._learning_system.get('policy_net')
        policy_optimizer = self._learning_system.get('policy_optimizer')
        value_net = self._learning_system.get('value_net')
        value_optimizer = self._learning_system.get('value_optimizer')

        algo_config = self._config.get('algo')
        max_steps = algo_config.get('max_steps')
        seg_len = algo_config.get('segment_length')
        ppo_opt_epochs = algo_config.get('ppo_opt_epochs')
        batch_size = algo_config.get('ppo_learner_batch_size')

        while self._learning_system.get('global_step') < max_steps:
            # generate trajectory.
            trajectory = self._trajectory_mgr.generate()
            metadata = trajectory.get('metadata')
            annotated = self._annotate(policy_net, value_net, trajectory)

            # update policy.
            for opt_epoch in range(ppo_opt_epochs):
                indices = np.random.permutation(seg_len)
                for i in range(0, seg_len, batch_size):
                    mb_indices = indices[i:i+batch_size]
                    mb = self._slice_minibatch(annotated, mb_indices)
                    losses = self._compute_losses(
                        mb=mb, policy_net=policy_net, value_net=value_net,
                        ent_coef=algo_config.get('ent_coef'),
                        clip_param=algo_config.get('clip_param')) # todo(lucaslingle): add support for annealing clipfrac.

                    policy_optimizer.zero_grad()
                    losses.get('composite_loss').backward()
                    policy_optimizer.step()

                    if value_net:
                        value_optimizer.zero_grad()
                        losses.get('composite_loss').backward()
                        value_optimizer.step()

            self._learning_system['global_step'] += seg_len
            global_metadata = global_means(metadata, self._config['world_size'])
            if self._rank == 0:
                for name in global_metadata:
                    self._writer.add_scalar(
                        tag=f"training/{name}",
                        scalar_value=global_metadata[name],
                        global_step=self._learning_system['global_step'])

    def evaluation_loop(self):
        raise NotImplementedError
