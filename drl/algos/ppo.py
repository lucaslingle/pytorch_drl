from contextlib import ExitStack

import torch as tc
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from drl.algos.abstract import Algo
from drl.algos.common import (
    TrajectoryManager, MultiDeque, extract_reward_name, gae_advantages,
    get_loss, global_means, global_gathers,
    update_trainable_wrappers, apply_pcgrad, pretty_print
)


class PPO(Algo):
    # PPO speed todos:
    #    after reproducability done, add support for vectorized environment.
    #    test speed of vectorized vs non-vectorized environments.
    def __init__(self, rank, config):
        super().__init__(rank, config)
        self._learning_system = self._get_learning_system()
        self._trajectory_mgr = TrajectoryManager(
            env=self._learning_system['env'],
            policy_net=self._learning_system['policy_net'],
            seg_len=self._config['algo']['seg_len'],
            extra_steps=self._config['algo']['gae_extra_steps'])
        self._metadata_acc = MultiDeque(
            memory_len=self._config['algo']['stats_memory_len'])
        if self._rank == 0:
            self._writer = SummaryWriter(self._config.get('log_dir'))

    def _get_learning_system(self):
        env_config = self._config.get('env')
        env = self._get_env(env_config)
        rank = self._rank

        policy_config = self._config['networks']['policy_net']
        policy_net = self._get_net(policy_config, env, rank)
        policy_optimizer_config = policy_config.get('optimizer')
        policy_optimizer = self._get_opt(policy_optimizer_config, policy_net)
        policy_scheduler_config = policy_config.get('scheduler')
        policy_scheduler = self._get_sched(
            policy_scheduler_config, policy_optimizer)

        value_config = self._config['networks']['value_net']
        value_net, value_optimizer, value_scheduler = None, None, None
        if not value_config.get('use_shared_architecture'):
            value_net = self._get_net(value_config, env, rank)
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

    def _get_reward_keys(self, omit_raw=True):
        reward_spec = self._learning_system['env'].reward_spec
        reward_keys = reward_spec.keys
        if omit_raw:
            reward_keys = [k for k in reward_keys if not k.endswith('_raw')]
            assert len(reward_keys) > 0
        return reward_keys

    def _maybe_split_prediction_keys(self, separate_value_net):
        reward_keys = self._get_reward_keys(omit_raw=True)
        policy_predict = ['policy']
        value_predict = [f'value_{k}' for k in reward_keys]
        if not separate_value_net:
            policy_predict.extend(value_predict)
        return policy_predict, value_predict

    def _slice_minibatch(self, trajectory, indices):
        results = dict()
        for field in trajectory:
            if isinstance(trajectory[field], dict):
                slice = {k: v[indices] for k,v in trajectory[field].items()}
            else:
                slice = trajectory[field][indices]
            results[field] = slice
        return results

    def _standardize(self, vector, eps=1e-8):
        vector -= tc.mean(vector)
        vector /= (tc.std(vector) + eps)
        return vector

    def _annotate(self, trajectory, policy_net, value_net, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            # decide who should predict what.
            separate_value_net = value_net is not None
            policy_predict, value_predict = self._maybe_split_prediction_keys(
                separate_value_net=separate_value_net)

            # make policy and value predictions.
            predictions = policy_net(
                trajectory['observations'], predict=policy_predict)
            pi = predictions['policy']

            if not separate_value_net:
                vpreds = {k: predictions[k] for k in value_predict}
            else:
                vpreds = value_net(
                    trajectory['observations'], predict=value_predict)

            # shallow copy of trajectory dict, point to initial/new values.
            trajectory_new = {
                'observations': trajectory['observations'],
                'actions': trajectory['actions'],
                'logprobs': pi.log_prob(trajectory['actions']),
                'entropies': pi.entropy()
            }
            trajectory_new = self._slice_minibatch(
                trajectory_new, slice(0, self._config['algo']['seg_len']))
            trajectory_new.update({
                'rewards': trajectory['rewards'],
                'dones': trajectory['dones'],
                'vpreds': {extract_reward_name(k): vpreds[k] for k in vpreds}
            })
            return trajectory_new

    @tc.no_grad()
    def _credit_assignment(self, trajectory):
        advantages, td_lambda_returns = dict(), dict()
        for k in self._get_reward_keys():
            advantages[k] = gae_advantages(
                seg_len=self._config['algo']['seg_len'],
                extra_steps=self._config['algo']['gae_extra_steps'],
                gamma=self._config['algo']['discount_gamma'][k],
                lam=self._config['algo']['gae_lambda'][k],
                rewards=trajectory['rewards'][k],
                vpreds=trajectory['vpreds'][k],
                dones=trajectory['dones'])
            td_lambda_returns[k] = advantages[k] + trajectory['vpreds'][k]
        trajectory.update({
            'advantages': advantages,
            'td_lambda_returns': td_lambda_returns
        })
        trajectory = self._slice_minibatch(
            trajectory, slice(0, self._config['algo']['seg_len']))
        trajectory = self._maybe_standardize_advantages(trajectory)
        return trajectory

    @tc.no_grad()
    def _maybe_standardize_advantages(self, trajectory):
        if self._config['algo']['standardize_adv']:
            for k in self._get_reward_keys():
                trajectory['advantages'][k] = self._standardize(
                    trajectory['advantages'][k])
        return trajectory

    def _get_reward_weightings(self):
        reward_weightings = {'extrinsic': 1.0}
        if hasattr(self._config['algo'], 'reward_weightings'):
            reward_weightings.update(self._config['algo']['reward_weightings'])
        assert set(reward_weightings.keys()) == set(self._get_reward_keys())
        return reward_weightings

    def _ppo_policy_entropy_bonus(self, mb_new, ent_coef, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            meanent = tc.mean(mb_new['entropies'])
            policy_entropy_bonus = ent_coef * meanent
            return {
                'meanent': meanent,
                'policy_entropy_bonus': policy_entropy_bonus
            }

    def _ppo_policy_surrogate_objective(self, mb_new, mb, clip_param, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            policy_surrogate_objective = tc.tensor(0.0)
            policy_ratios = tc.exp(mb_new['logprobs'] - mb['logprobs'])
            clipped_policy_ratios = tc.clip(
                policy_ratios, 1-clip_param, 1+clip_param)
            reward_weightings = self._get_reward_weightings()
            for key in self._get_reward_keys():
                surr1 = mb['advantages'][key] * policy_ratios
                surr2 = mb['advantages'][key] * clipped_policy_ratios
                pol_surr_for_rew = tc.mean(tc.min(surr1, surr2))
                if key == 'extrinsic':
                    clipfrac = tc.mean(tc.greater(surr1, surr2).float())
                weight = reward_weightings[key]
                policy_surrogate_objective += weight * pol_surr_for_rew
            return {
                'policy_surrogate_objective': policy_surrogate_objective,
                'clipfrac': clipfrac
            }

    def _ppo_vf_loss(self, mb_new, mb, clip_param, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            vf_loss = tc.tensor(0.0)
            vf_clipfrac = tc.tensor(0.0)
            vf_loss_criterion = get_loss(self._config['algo']['vf_loss_cls'])
            vf_loss_clipping = self._config['algo']['vf_loss_clipping']
            reward_weightings = self._get_reward_weightings()
            for key in self._get_reward_keys():
                tdlam_rets = mb['td_lambda_returns'][key]
                vpreds = mb['vpreds'][key]
                vpreds_new = mb_new['vpreds'][key]
                if vf_loss_clipping:
                    vpreds_new_clipped = tc.clip(
                        vpreds_new, vpreds-clip_param, vpreds+clip_param)
                    vsurr1 = vf_loss_criterion(
                        input=vpreds_new, target=tdlam_rets)
                    vsurr2 = vf_loss_criterion(
                        input=vpreds_new_clipped, target=tdlam_rets)
                    vf_loss_for_rew = tc.mean(tc.max(vsurr1, vsurr2))
                    if key == 'extrinsic':
                        vf_clipfrac = tc.mean(tc.less(vsurr1, vsurr2).float())
                else:
                    vf_loss_for_rew = tc.mean(vf_loss_criterion(
                        input=vpreds_new, target=tdlam_rets))
                weight = reward_weightings[key]
                vf_loss += (weight ** 2) * vf_loss_for_rew
                # todo(lucaslingle): support other vf loss weightings
                #    besides implicit mse loss weighting used here.
            return {
                'vf_loss': vf_loss,
                'vf_clipfrac': vf_clipfrac
            }

    def _compute_losses(self, mb, policy_net, value_net, clip_param, ent_coef, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            mb_new = self._annotate(mb, policy_net, value_net, no_grad=no_grad)

            entropy_quantities = self._ppo_policy_entropy_bonus(
                mb_new=mb_new, ent_coef=ent_coef, no_grad=no_grad)

            policy_quantities = self._ppo_policy_surrogate_objective(
                mb_new=mb_new, mb=mb, clip_param=clip_param, no_grad=no_grad)

            value_quantities = self._ppo_vf_loss(
                mb_new=mb_new, mb=mb, clip_param=clip_param, no_grad=no_grad)

            policy_objective = policy_quantities['policy_surrogate_objective']
            policy_objective += entropy_quantities['policy_entropy_bonus']
            policy_loss = -policy_objective

            value_loss = value_quantities['vf_loss']

            separate_value_net = value_net is not None
            vf_loss_coef = self._config['algo']['vf_loss_coef']
            vf_weight = 1. if separate_value_net else vf_loss_coef
            composite_loss = policy_loss + vf_weight * value_loss

            return {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'composite_loss': composite_loss,
                'meanent': entropy_quantities['meanent'],
                'clipfrac': policy_quantities['clipfrac'],
                'vf_clipfrac': value_quantities['vf_clipfrac']
            }

    def _pcgrad_checks(self):
        if len(self._config['networks']['policy_net']['predictors']) <= 1:
            msg = "Required multiple predictions for pcgrad"
            raise ValueError(msg)
        if self._learning_system['value_net'] is not None:
            msg = "Currently only support pcgrad when no val net"
            raise ValueError(msg)

    def _maybe_split_losses(self, losses, value_net, use_pcgrad):
        policy_losses = dict()
        value_losses = dict()
        for k in losses:
            if k.startswith('policy_'):
                policy_losses[k] = losses[k]
            if k.startswith('value_'):
                if value_net:
                    value_losses[k] = losses[k]
                else:
                    policy_losses[k] = losses[k]
            if not use_pcgrad:
                if k == 'composite_loss':
                    policy_losses[k] = losses[k]
        return policy_losses, value_losses

    def _optimize_losses(self, net, optimizer, losses, retain_graph, use_pcgrad):
        if not use_pcgrad:
            optimizer.zero_grad()
            losses['composite_loss'].backward(retain_graph=retain_graph)
            optimizer.step()
        else:
            apply_pcgrad(
                network=net,
                optimizer=optimizer,
                task_losses=losses,
                normalize=False)
            optimizer.step()

    def training_loop(self):
        world_size = self._config['distributed']['world_size']

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
        seg_len = algo_config.get('seg_len')
        opt_epochs = algo_config.get('opt_epochs')
        batch_size = algo_config.get('learner_batch_size')
        checkpoint_frequency = algo_config.get('checkpoint_frequency')
        use_pcgrad = algo_config.get('use_pcgrad')
        if use_pcgrad:
            self._pcgrad_checks()
        separate_value_net = value_net is not None

        while self._learning_system.get('global_step') < max_steps:
            # generate trajectory.
            trajectory = self._trajectory_mgr.generate()
            metadata = trajectory.pop('metadata')
            trajectory = self._annotate(
                trajectory, policy_net, value_net, no_grad=True)
            trajectory = self._credit_assignment(trajectory)
            self._learning_system['global_step'] += seg_len * world_size

            # update policy.
            for opt_epoch in range(opt_epochs):
                indices = np.random.permutation(seg_len)
                for i in range(0, seg_len, batch_size):
                    mb_indices = indices[i:i+batch_size]
                    mb = self._slice_minibatch(trajectory, mb_indices)
                    losses = self._compute_losses(
                        mb=mb, policy_net=policy_net, value_net=value_net,
                        ent_coef=ent_coef_annealer.value,
                        clip_param=clip_param_annealer.value,
                        no_grad=False)
                    policy_losses, value_losses = self._maybe_split_losses(
                        losses=losses, value_net=separate_value_net,
                        use_pcgrad=use_pcgrad)
                    self._optimize_losses(
                        net=policy_net, optimizer=policy_optimizer,
                        losses=policy_losses, retain_graph=separate_value_net,
                        use_pcgrad=use_pcgrad)
                    if separate_value_net:
                        self._optimize_losses(
                            net=value_net, optimizer=value_optimizer,
                            losses=value_losses, retain_graph=False,
                            use_pcgrad=use_pcgrad)
                    update_trainable_wrappers(env, mb)

                metrics = self._compute_losses(
                    mb=trajectory, policy_net=policy_net, value_net=value_net,
                    ent_coef=ent_coef_annealer.value,
                    clip_param=clip_param_annealer.value,
                    no_grad=True)
                global_metrics = global_means(metrics, world_size, item=True)
                if self._rank == 0:
                    print(f"Opt epoch: {opt_epoch}")
                    pretty_print(global_metrics)
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
            global_metadata = global_gathers(metadata, world_size)
            self._metadata_acc.update(global_metadata)
            if self._rank == 0:
                pretty_print(self._metadata_acc)
                for name in global_metadata:
                    self._writer.add_scalar(
                        tag=f"metadata/{name}",
                        scalar_value=self._metadata_acc.mean(name),
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

    @tc.no_grad()
    def render_loop(self):
        # todo: replace this with video-saving loop in Algo abstract class.
        #   or implement saving video via wrapper env.
        if self._rank == 0:
            env = self._learning_system['env']
            policy_net = self._learning_system['policy_net']
            t = 0
            r_tot = 0.
            o_t = env.reset()
            real_done_t = False
            while not real_done_t:
                predictions_t = policy_net(
                    x=tc.tensor(o_t).float().unsqueeze(0),
                    predict=['policy'])
                pi_dist_t = predictions_t['policy']
                a_t = pi_dist_t.sample().squeeze(0).detach().numpy()
                o_tp1, r_t, done_t, info_t = env.step(a_t)
                if 'ale.lives' in info_t:
                    real_done_t = (info_t['ale.lives'] == 0)
                else:
                    real_done_t = done_t
                _ = env.render(mode='human')
                t += 1
                r_tot += r_t['extrinsic_raw']
                o_t = o_tp1
            print(r_tot)


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
