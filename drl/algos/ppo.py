from contextlib import ExitStack

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
            seg_len=self._config['algo']['segment_length'],
            extra_steps=0)
        if self._rank == 0:
            self._writer = SummaryWriter(self._config.get('log_dir'))

    def _get_learning_system(self):
        env_config = self._config.get('env')
        env = self._get_env(env_config)

        policy_config = self._config['networks']['policy_net']
        policy_net = self._get_net(policy_config, env)
        policy_optimizer_config = policy_config.get('optimizer')
        policy_optimizer = self._get_opt(policy_optimizer_config, policy_net)
        policy_scheduler_config = policy_config.get('scheduler')
        policy_scheduler = self._get_sched(
            policy_scheduler_config, policy_optimizer)

        value_config = self._config['networks']['value_net']
        value_net, value_optimizer, value_scheduler = None, None, None
        if not value_config.get('use_shared_architecture'):
            value_net = self._get_net(value_config, env)
            value_optimizer_config = value_config.get('optimizer')
            value_optimizer = self._get_opt(value_optimizer_config, value_net)
            value_scheduler_config = value_config.get('scheduler')
            value_scheduler = self._get_sched(
                value_scheduler_config, value_optimizer)

        algo_config = self._config.get('algo')
        max_steps = algo_config.get('max_steps')
        seg_len = algo_config.get('seg_len')
        num_policy_improvements = max_steps // seg_len
        clip_param_annealer = Annealer(
            initial_value=algo_config.get('clip_param_init'),
            final_value=algo_config.get('clip_param_final'),
            num_policy_improvements=num_policy_improvements)
        ent_coef_annealer = Annealer(
            initial_value=algo_config.get('ent_coef_init'),
            final_value=algo_config.get('ent_coef_final'),
            num_policy_improvements=num_policy_improvements)

        checkpointables = {
            'policy_net': policy_net,
            'policy_optimizer': policy_optimizer,
            'policy_scheduler': policy_scheduler,
            'value_net': value_net,
            'value_optimizer': value_optimizer,
            'value_scheduler': value_scheduler,
            'clip_param_annealer': clip_param_annealer,
            'ent_coef_annealer': ent_coef_annealer
        }
        checkpointables_ = {k: v for k,v in checkpointables.items()}
        checkpointables_.update(env.get_checkpointables())
        global_step = self._maybe_load_checkpoints(checkpointables_, step=None)
        return {'global_step': global_step, 'env': env, **checkpointables}

    def _slice_minibatch(self, trajectory, indices):
        results = dict()
        for field in trajectory:
            if isinstance(trajectory[field], dict):
                slice = {k: v[indices] for k,v in trajectory[field].items()}
            else:
                slice = trajectory[field][indices]
            results[field] = slice
        return results

    def _annotate(self, trajectory, policy_net, value_net, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            # get config variables.
            algo_config = self._config.get('algo')
            seg_len = algo_config.get('segment_length')

            # get trajectory variables.
            observations = trajectory.get('observations')
            actions = trajectory.get('actions')
            rewards = trajectory.get('rewards')

            # decide who should predict what.
            reward_keys = rewards.keys()
            relevant_reward_keys = [k for k in reward_keys if k != 'extrinsic_raw']
            policy_predict = ['policy']
            value_predict = [f'value_{k}' for k in relevant_reward_keys]
            if value_net is None:
                policy_predict.extend(value_predict)

            # compute logprobs of actions.
            predictions = policy_net(observations, policy_predict)
            pi = predictions.get('policy')
            logprobs = pi.log_prob(actions)
            entropies = pi.entropy()

            # compute value estimates.
            if value_net is None:
                vpreds = {k: predictions[k] for k in value_predict}
            else:
                vpreds = value_net(observations, value_predict)
            vpreds = {k.partition('_')[2]: vpreds[k] for k in vpreds}

            trajectory.update({'logprobs': logprobs, 'entropies': entropies})
            trajectory = self._slice_minibatch(trajectory, slice(0, seg_len))
            trajectory.update({'vpreds': vpreds})
            return trajectory

    @tc.no_grad()
    def _credit_assignment(self, trajectory):
        # get config variables.
        algo_config = self._config.get('algo')
        seg_len = algo_config.get('segment_length')
        lam = algo_config.get('gae_lambda')
        gamma = algo_config.get('discount_gamma')

        # get trajectory variables.
        rewards = trajectory.get('rewards')
        dones = trajectory.get('dones')
        vpreds = trajectory.get('vpreds')

        # get reward keys.
        reward_keys = rewards.keys()
        relevant_reward_keys = [k for k in reward_keys if k != 'extrinsic_raw']

        # assign credit.
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
                delta_t = -V_t + r_t + (1.-dones[t]) * gamma[k] * V_tp1
                A_t = delta_t + (1.-dones[t]) * gamma[k] * lam[k] * A_tp1
                advantages[k][t] = A_t
        td_lambda_returns = {k: advantages[k] + vpreds[k] for k in advantages}
        trajectory.update({
            'advantages': advantages,
            'td_lambda_returns': td_lambda_returns
        })
        return self._slice_minibatch(trajectory, slice(0, seg_len))

    def _compute_losses(self, mb, policy_net, value_net, clip_param, ent_coef):
        mb_new = self._annotate(mb, policy_net, value_net, no_grad=False)

        # get reward keys.
        reward_keys = mb_new['rewards'].keys()
        relevant_reward_keys = [k for k in reward_keys if k != 'extrinsic_raw']

        # entropy
        entropies = mb_new['entropies']
        mean_entropy = tc.mean(entropies)
        policy_entropy_bonus = ent_coef * mean_entropy

        # ppo policy loss, value loss
        policy_ratio = tc.exp(mb_new['logprobs'] - mb['logprobs'])
        clipped_policy_ratio = tc.clip(policy_ratio, 1-clip_param, 1+clip_param)
        policy_surrogate_objective = 0.
        vf_loss = 0.
        clipfrac = 0.
        for key in relevant_reward_keys:
            surr1 = mb['advantages'][key] * policy_ratio
            surr2 = mb['advantages'][key] * clipped_policy_ratio
            ppo_surr_for_reward = tc.mean(tc.min(surr1, surr2))
            vf_loss_for_reward = tc.mean(
                tc.square(mb['td_lambda_returns'][key] - mb_new['vpreds'][key])
            )
            clipfrac_for_reward = tc.mean(tc.greater(surr1, surr2).float())
            if len(relevant_reward_keys) == 1:
                policy_surrogate_objective += ppo_surr_for_reward
                vf_loss += vf_loss_for_reward
            else:
                weight = self._config['algo']['reward_weightings'][key]
                policy_surrogate_objective += weight * ppo_surr_for_reward
                vf_loss += tc.square(weight) * vf_loss_for_reward
                # todo(lucaslingle): investigate if official implementation
                #    of RND uses this value loss weighting
                # todo(lucaslingle): add support for configuring different
                #    value losses, such as huber instead of squared
                # todo(lucaslingle): add suppport for configuring
                #    clipped or non-clipped value losses.
            clipfrac += clipfrac_for_reward

        policy_loss = -(policy_surrogate_objective + policy_entropy_bonus)
        weight = 1. if value_net else self._config['algo']['vf_loss_coef']
        composite_loss = policy_loss + weight * vf_loss
        return {
            'policy_loss': policy_loss,
            'value_loss': vf_loss,
            'composite_loss': composite_loss,
            'meanent': mean_entropy,
            'clipfrac': clipfrac / len(relevant_reward_keys)
        }

    def training_loop(self):
        world_size = self._config.get('world_size')

        env = self._learning_system.get('env')
        policy_net = self._learning_system.get('policy_net')
        policy_optimizer = self._learning_system.get('policy_optimizer')
        policy_scheduler = self._learning_system.get('policy_scheduler')
        value_net = self._learning_system.get('value_net')
        value_optimizer = self._learning_system.get('value_optimizer')
        value_scheduler = self._learning_system.get('value_scheduler')
        clip_param_annealer = self._learning_system.get('clip_param_annealer')
        ent_coef_annealer = self._learning_system.get('ent_coef_annealer')

        algo_config = self._config.get('algo')
        max_steps = algo_config.get('max_steps')
        seg_len = algo_config.get('segment_length')
        opt_epochs = algo_config.get('opt_epochs')
        batch_size = algo_config.get('learner_batch_size')
        checkpoint_frequency = algo_config.get('checkpoint_frequency')

        while self._learning_system.get('global_step') < max_steps:
            # generate trajectory.
            trajectory = self._trajectory_mgr.generate()
            metadata = trajectory.pop('metadata')
            trajectory = self._annotate(
                trajectory, policy_net, value_net, no_grad=True)
            trajectory = self._credit_assignment(trajectory)
            self._learning_system['global_step'] += seg_len

            # update policy.
            for opt_epoch in range(opt_epochs):
                indices = np.random.permutation(seg_len)
                for i in range(0, seg_len, batch_size):
                    mb_indices = indices[i:i+batch_size]
                    mb = self._slice_minibatch(trajectory, mb_indices)
                    losses = self._compute_losses(
                        mb=mb, policy_net=policy_net, value_net=value_net,
                        ent_coef=ent_coef_annealer.value,
                        clip_param=clip_param_annealer.value)
                    composite_loss = losses.get('composite_loss')

                    policy_optimizer.zero_grad()
                    composite_loss.backward(retain_graph=value_net is not None)
                    policy_optimizer.step()

                    if value_net:
                        value_optimizer.zero_grad()
                        composite_loss.backward()
                        value_optimizer.step()

                    global_metrics = global_means(losses, world_size)
                    if self._rank == 0:
                        for name in global_metrics:
                            self._writer.add_scalar(
                                tag=f"epoch_{opt_epoch}/{name}",
                                scalar_value=global_metrics[name],
                                global_step=self._learning_system['global_step'])

            if policy_scheduler:
                policy_scheduler.step()
            if value_scheduler:
                value_scheduler.step()
            clip_param_annealer.step()
            ent_coef_annealer.step()

            # save everything.
            global_metadata = global_means(metadata, world_size)
            if self._rank == 0:
                for name in global_metadata:
                    self._writer.add_scalar(
                        tag=f"metadata/{name}",
                        scalar_value=global_metadata[name],
                        global_step=self._learning_system['global_step'])

                global_step = self._learning_system['global_step']
                if (global_step // seg_len) % checkpoint_frequency == 0:
                    self._save_checkpoints(
                        checkpointables={
                            'policy_net': policy_net,
                            'policy_optimizer': policy_optimizer,
                            'policy_scheduler': policy_scheduler,
                            'value_net': value_net,
                            'value_optimizer': value_optimizer,
                            'value_scheduler': value_scheduler,
                            'clip_param_annealer': clip_param_annealer,
                            'ent_coef_annealer': ent_coef_annealer,
                            **env.get_checkpointables()
                        },
                        step=global_step)

    def evaluation_loop(self):
        raise NotImplementedError


class Annealer(tc.nn.Module):
    def __init__(
            self, initial_value, final_value, num_policy_improvements
    ):
        super().__init__()
        self._initial_value = initial_value
        self._final_value = final_value
        self._num_policy_improvements = num_policy_improvements
        self.register_buffer('_num_steps', tc.tensor(0))

    @property
    def num_steps(self):
        return self._num_steps.item()

    @property
    def value(self):
        frac_done = self.num_steps / self._num_policy_improvements
        s = min(max(0., frac_done), 1.)
        return self._initial_value * (1. - s) + self._final_value * s

    def step(self):
        self.register_buffer('_num_steps', tc.tensor(self.num_steps+1))
