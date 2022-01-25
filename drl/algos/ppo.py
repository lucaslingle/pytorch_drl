from typing import Mapping, Any, Union, Optional
from contextlib import ExitStack

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym

from drl.algos.abstract import Algo
from drl.algos.common import (
    TrajectoryManager, MultiDeque, extract_reward_name,
    get_credit_assignment_ops, get_loss, global_means, global_gathers,
    update_trainable_wrappers, apply_pcgrad, pretty_print
)
from drl.envs.wrappers import Wrapper
from drl.agents.integration import Agent
from drl.utils.checkpointing import save_checkpoints


class PPO(Algo):
    # PPO speed todos:
    #    after reproducability done, add support for vectorized environment.
    #    test speed of vectorized vs non-vectorized environments.
    def __init__(
            self,
            rank: int,
            world_size: int,
            seg_len: int,
            opt_epochs: int,
            learner_batch_size: int,
            clip_param_init: float,
            clip_param_final: float,
            ent_coef_init: float,
            ent_coef_final: float,
            vf_loss_cls: str,
            vf_loss_coef: float,
            vf_loss_clipping: bool,
            vf_simple_weighting: bool,
            credit_assignment_spec: Mapping[str, Mapping[str, Union[str, Mapping[str, Any]]]],
            extra_steps: int,
            standardize_adv: bool,
            use_pcgrad: bool,
            stats_memory_len: int,
            checkpoint_frequency: int,
            non_learning_steps: int,
            max_steps: int,
            global_step: int,
            env: Union[gym.core.Env, Wrapper],
            policy_net: Union[Agent, DDP],
            policy_optimizer: tc.optim.Optimizer,
            policy_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],
            value_net: Optional[Union[Agent, DDP]],
            value_optimizer: Optional[tc.optim.Optimizer],
            value_scheduler: Optional[tc.optim.lr_scheduler._LRScheduler],
            log_dir: str,
            checkpoint_dir: str,
            media_dir: str,
            reward_weights: Optional[Mapping[str, float]] = None
    ):
        """
        Args:
            rank: Process rank.
            world_size: Total number of processes.
            seg_len: Trajectory segment length.
            opt_epochs: Optimization epochs per policy improvement phase in PPO.
            learner_batch_size: Batch size per learner process.
            clip_param_init: Initial PPO clip parameter.
            clip_param_final: Final PPO clip parameter, to which clip_param_init
                will be linearly annealed.
            ent_coef_init: Initial entropy bonus coefficient.
            ent_coef_final: Final entropy bonus coefficient, to which
                ent_coef_init will be linearly annealed.
            vf_loss_cls: Value function loss class name. Name must match
                a derived class of _Loss in torch.nn.modules.loss.
                The most useful classes are MSELoss and SmoothL1Loss.
            vf_loss_coef: Value function loss coefficient.
                Ignored if value network is separate from policy network.
            vf_loss_clipping: If true, use pessimistic value function loss.
            vf_simple_weighting: If true, use equal weighting of all value
                function losses. Ignored if env.reward_spec.keys() does not
                contain any intrinsic rewards.
            credit_assignment_spec: Mapping from reward names to
                advantage estimator classes and their arguments.
            extra_steps: Extra steps required for credit assignment.
                Should be set to n-1 if using n-step return-based advantage
                estimation.
            standardize_adv: Standardize advantages per trajectory segment?
            use_pcgrad: Use the PCGrad algorithm from Yu et al., 2020?
                Only allowed if policy and value architecture is shared.
            stats_memory_len: Window size for moving average of episode metadata.
            checkpoint_frequency: Checkpoint frequency, measured in global steps.
            non_learning_steps: Number of global steps to skip integration learning.
                Useful in conjunction with wrappers that maintain rolling statistics.
            max_steps: Maximum number of global steps.
            global_step: Global step of learning so far.
            env: Environment instance or wrapped environment.
            policy_net: Agent instance or DDP-wrapped Agent instance.
                Must have 'policy' as a prediction key.
            policy_optimizer: Optimizer for policy_net.
            policy_scheduler: Optional learning rate scheduler for
                policy_optimizer.
            value_net: Optional Agent instance, DDP-wrapped Agent instance.
                If not None, must have a 'value_{reward_name}' prediction key
                for each reward_name in env.reward_spec.keys()
                other than 'extrinsic_raw'.
            value_optimizer: Optional Optimizer for value_net.
                Required if value_net is not None.
            value_scheduler: Optional learning rate scheduler for
                value_optimizer.
            checkpoint_dir: Checkpoint directory.
            log_dir: Tensorboard logs directory.
            media_dir: Media directory.
            reward_weights: Optional reward weights mapping,
                keyed by reward name. Ignored if env.reward_spec.keys()
                does not contain any intrinsic rewards. Required if it does.
                Default value is None.
        """
        super().__init__(rank)
        self._world_size = world_size
        self._seg_len = seg_len
        self._opt_epochs = opt_epochs
        self._learner_batch_size = learner_batch_size
        self._clip_param = Annealer(
            clip_param_init, clip_param_final, max_steps)
        self._ent_coef = Annealer(
            ent_coef_init, ent_coef_final, max_steps)
        self._vf_loss_criterion = get_loss(vf_loss_cls)
        self._vf_loss_coef = vf_loss_coef
        self._vf_loss_clipping = vf_loss_clipping
        self._vf_simple_weighting = vf_simple_weighting
        self._credit_assignment_ops = get_credit_assignment_ops(
            seg_len, extra_steps, credit_assignment_spec)
        self._extra_steps = extra_steps
        self._standardize_adv = standardize_adv
        self._use_pcgrad = use_pcgrad
        self._reward_weights = reward_weights

        self._stats_memory_len = stats_memory_len
        self._checkpoint_frequency = checkpoint_frequency
        self._non_learning_steps = non_learning_steps
        self._max_steps = max_steps

        self._global_step = global_step
        self._env = env
        self._policy_net = policy_net
        self._policy_optimizer = policy_optimizer
        self._policy_scheduler = policy_scheduler
        self._value_net = value_net
        self._value_optimizer = value_optimizer
        self._value_scheduler = value_scheduler

        self._checkpoint_dir = checkpoint_dir
        self._log_dir = log_dir
        self._media_dir = media_dir

        self._trajectory_mgr = TrajectoryManager(
            env=env, policy_net=policy_net, seg_len=seg_len,
            extra_steps=extra_steps)
        self._metadata_acc = MultiDeque(memory_len=stats_memory_len)
        if self._rank == 0:
            self._writer = SummaryWriter(log_dir)

    def _get_reward_keys(self, omit_raw=True):
        reward_spec = self._env.reward_spec
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
            # make policy and value predictions.
            separate_value_net = value_net is not None
            policy_predict, value_predict = self._maybe_split_prediction_keys(
                separate_value_net=separate_value_net)
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
                trajectory_new, slice(0, self._seg_len))
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
            advantages[k] = self._credit_assignment_ops[k].estimate_advantages(
                rewards=trajectory['rewards'][k],
                vpreds=trajectory['vpreds'][k],
                dones=trajectory['dones'])
            td_lambda_returns[k] = advantages[k] + trajectory['vpreds'][k]
        trajectory.update({
            'advantages': advantages,
            'td_lambda_returns': td_lambda_returns
        })
        trajectory = self._slice_minibatch(
            trajectory, slice(0, self._seg_len))
        trajectory = self._maybe_standardize_advantages(trajectory)
        return trajectory

    @tc.no_grad()
    def _maybe_standardize_advantages(self, trajectory):
        if self._standardize_adv:
            for k in self._get_reward_keys():
                trajectory['advantages'][k] = self._standardize(
                    trajectory['advantages'][k])
        return trajectory

    def _get_reward_weightings(self):
        reward_weightings = {'extrinsic': 1.0}
        if self._reward_weights is not None:
            reward_weightings.update(self._reward_weights)
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
            policy_ratios = tc.exp(mb_new['logprobs'] - mb['logprobs'])
            clipped_policy_ratios = tc.clip(policy_ratios, 1-clip_param, 1+clip_param)
            reward_weightings = self._get_reward_weightings()
            advantage_shape = mb['advantages']['extrinsic'].shape
            advantages = tc.zeros(advantage_shape, dtype=tc.float32)
            for key in self._get_reward_keys():
                advantages += reward_weightings[key] * mb['advantages'][key]
            surr1 = advantages * policy_ratios
            surr2 = advantages * clipped_policy_ratios
            policy_surrogate_objective = tc.mean(tc.min(surr1, surr2))
            clipfrac = tc.mean(tc.greater(surr1, surr2).float())
            return {
                'policy_surrogate_objective': policy_surrogate_objective,
                'clipfrac': clipfrac
            }

    def _ppo_vf_loss(self, mb_new, mb, clip_param, no_grad):
        with tc.no_grad() if no_grad else ExitStack():
            vf_loss = tc.tensor(0.0)
            reward_weightings = self._get_reward_weightings()
            for key in self._get_reward_keys():
                tdlam_rets = mb['td_lambda_returns'][key]
                vpreds = mb['vpreds'][key]
                vpreds_new = mb_new['vpreds'][key]
                if self._vf_loss_clipping:
                    vpreds_new_clipped = tc.clip(
                        vpreds_new, vpreds-clip_param, vpreds+clip_param)
                    vsurr1 = self._vf_loss_criterion(
                        input=vpreds_new, target=tdlam_rets)
                    vsurr2 = self._vf_loss_criterion(
                        input=vpreds_new_clipped, target=tdlam_rets)
                    vf_loss_for_rew = tc.mean(tc.max(vsurr1, vsurr2))
                else:
                    vf_loss_for_rew = tc.mean(self._vf_loss_criterion(
                        input=vpreds_new, target=tdlam_rets))
                if self._vf_simple_weighting:
                    vf_loss += vf_loss_for_rew
                else:
                    vf_loss += (reward_weightings[key] ** 2) * vf_loss_for_rew
            return {
                'vf_loss': vf_loss
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
            vf_loss_coef = self._vf_loss_coef
            vf_weight = 1. if separate_value_net else vf_loss_coef
            composite_loss = policy_loss + vf_weight * value_loss
            return {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'composite_loss': composite_loss,
                'meanent': entropy_quantities['meanent'],
                'clipfrac': policy_quantities['clipfrac']
            }

    def _pcgrad_checks(self):
        if len(self._policy_net.keys) <= 1:
            msg = "Required multiple predictions for pcgrad"
            raise ValueError(msg)
        if self._value_net is not None:
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
                    value_losses[k] = losses[k]
        return policy_losses, value_losses

    def _optimize_losses(self, net, optimizer, losses, retain_graph, use_pcgrad):
        optimizer.zero_grad()
        if not use_pcgrad:
            losses['composite_loss'].backward(retain_graph=retain_graph)
        else:
            apply_pcgrad(
                network=net,
                optimizer=optimizer,
                task_losses=losses,
                normalize=False)
        optimizer.step()

    def training_loop(self):
        if self._use_pcgrad:
            self._pcgrad_checks()
        separate_value_net = self._value_net is not None

        while self._global_step < self._max_steps:
            # generate trajectory.
            trajectory = self._trajectory_mgr.generate()
            metadata = trajectory.pop('metadata')
            trajectory = self._annotate(
                trajectory, self._policy_net, self._value_net, no_grad=True)
            trajectory = self._credit_assignment(trajectory)
            self._global_step += self._seg_len * self._world_size

            # update policy.
            for opt_epoch in range(self._opt_epochs):
                indices = np.random.permutation(self._seg_len)
                for i in range(0, self._seg_len, self._learner_batch_size):
                    mb_indices = indices[i:i+self._learner_batch_size]
                    mb = self._slice_minibatch(trajectory, mb_indices)
                    update_trainable_wrappers(self._env, mb)
                    if self._global_step <= self._non_learning_steps:
                        continue
                    losses = self._compute_losses(
                        mb=mb, policy_net=self._policy_net, value_net=self._value_net,
                        ent_coef=self._ent_coef.value(self._global_step),
                        clip_param=self._clip_param.value(self._global_step),
                        no_grad=False)
                    policy_losses, value_losses = self._maybe_split_losses(
                        losses=losses, value_net=separate_value_net,
                        use_pcgrad=self._use_pcgrad)
                    self._optimize_losses(
                        net=self._policy_net, optimizer=self._policy_optimizer,
                        losses=policy_losses, retain_graph=separate_value_net,
                        use_pcgrad=self._use_pcgrad)
                    if separate_value_net:
                        self._optimize_losses(
                            net=self._value_net, optimizer=self._value_optimizer,
                            losses=value_losses, retain_graph=False,
                            use_pcgrad=self._use_pcgrad)

                metrics = self._compute_losses(
                    mb=trajectory, policy_net=self._policy_net, value_net=self._value_net,
                    ent_coef=self._ent_coef.value(self._global_step),
                    clip_param=self._clip_param.value(self._global_step),
                    no_grad=True)
                global_metrics = global_means(metrics, self._world_size, item=True)
                if self._rank == 0:
                    print(f"Opt epoch: {opt_epoch}")
                    pretty_print(global_metrics)
                    for name in global_metrics:
                        self._writer.add_scalar(
                            tag=f"epoch_{opt_epoch}/{name}",
                            scalar_value=global_metrics[name],
                            global_step=self._global_step)

            if self._policy_scheduler:
                self._policy_scheduler.step()
            if self._value_scheduler:
                self._value_scheduler.step()

            # save everything.
            global_metadata = global_gathers(metadata, self._world_size)
            self._metadata_acc.update(global_metadata)
            if self._rank == 0:
                pretty_print(self._metadata_acc)
                for name in global_metadata:
                    self._writer.add_scalar(
                        tag=f"metadata/{name}",
                        scalar_value=self._metadata_acc.mean(name),
                        global_step=self._global_step)

                if self._global_step % self._checkpoint_frequency == 0:
                    save_checkpoints(
                        checkpoint_dir=self._checkpoint_dir,
                        checkpointables={
                            'policy_net': self._policy_net,
                            'policy_optimizer': self._policy_optimizer,
                            'policy_scheduler': self._policy_scheduler,
                            'value_net': self._value_net,
                            'value_optimizer': self._value_optimizer,
                            'value_scheduler': self._value_scheduler,
                            **self._env.get_checkpointables()
                        },
                        steps=self._global_step)

    def evaluation_loop(self):
        raise NotImplementedError

    @tc.no_grad()
    def render_loop(self):
        # todo: replace this with video-saving loop in Algo abstract class.
        #   or implement saving video via wrapper env.
        if self._rank == 0:
            t = 0
            r_tot = 0.
            o_t = self._env.reset()
            done_t = False
            while not done_t:
                predictions_t = self._policy_net(
                    x=tc.tensor(o_t).float().unsqueeze(0),
                    predict=['policy'])
                pi_dist_t = predictions_t['policy']
                a_t = pi_dist_t.sample().squeeze(0).detach().numpy()
                o_tp1, r_t, done_t, info_t = self._env.step(a_t)
                _ = self._env.render(mode='human')
                t += 1
                r_tot += r_t['extrinsic_raw']
                o_t = o_tp1
            print(r_tot)


class Annealer:
    def __init__(self, initial_value, final_value, max_steps):
        self._initial_value = initial_value
        self._final_value = final_value
        self._max_steps = max_steps

    def value(self, global_step):
        frac_done = global_step / self._max_steps
        s = min(max(0., frac_done), 1.)
        return self._initial_value * (1. - s) + self._final_value * s
