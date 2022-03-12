from typing import Mapping, Union, Optional, Dict, Tuple, Any
from contextlib import ExitStack

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
import gym

from drl.algos.abstract import ActorCriticAlgo
from drl.algos.common import (
    AdvantageEstimator,
    get_loss,
    global_means,
    global_gathers,
    update_trainable_wrappers,
    apply_pcgrad,
    pretty_print,
    LinearSchedule,
    IndependentSampler)
from drl.envs.wrappers import Wrapper
from drl.utils.checkpointing import save_checkpoints
from drl.utils.nested import slice_nested_tensor
from drl.utils.types import NestedTensor, Optimizer, Scheduler


class PPO(ActorCriticAlgo):
    """
    Proximal Policy Optimization (clip variant).

    Reference:
        J. Schulman et al., 2017 -
            'Proximal Policy Optimization Algorithms'.
    """
    def __init__(
            self,
            rank: int,
            world_size: int,
            rollout_len: int,
            extra_steps: int,
            credit_assignment_ops: Mapping[str, AdvantageEstimator],
            stats_window_len: int,
            non_learning_steps: int,
            max_steps: int,
            checkpoint_frequency: int,
            checkpoint_dir: str,
            log_dir: str,
            media_dir: str,
            global_step: int,
            env: Union[gym.core.Env, Wrapper],
            policy_net: DDP,
            policy_optimizer: Optimizer,
            policy_scheduler: Optional[Scheduler],
            value_net: Optional[DDP],
            value_optimizer: Optional[Optimizer],
            value_scheduler: Optional[Scheduler],
            standardize_adv: bool,
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
            use_pcgrad: bool,
            reward_weights: Optional[Mapping[str, float]] = None) -> None:
        """
        Args:
            rank (int): Process rank.
            world_size (int): Total number of processes.
            rollout_len (int): Trajectory segment length.
            extra_steps (int): Extra steps required for credit assignment.
                Should be set to n-1 if using n-step return-based advantage
                estimation.
            credit_assignment_ops (Mapping[str, AdvantageEstimator]):
                Mapping from reward names to AdvantageEstimator instances.
            stats_window_len (int): Window size for moving average of episode
                 metadata.
            non_learning_steps (int): Number of global steps to skip agent
                learning. Useful in conjunction with wrappers that maintain
                rolling statistics.
            max_steps (int): Maximum number of global steps.
            checkpoint_frequency (int): Checkpoint frequency, measured in
                global steps.
            checkpoint_dir (str): Checkpoint directory.
            log_dir (str): Tensorboard logs directory.
            media_dir (str): Media directory.
            global_step (int): Global step of learning so far.
            env (Union[gym.core.Env, Wrapper]): Environment instance or wrapped
                environment.
            policy_net (torch.nn.parallel.DistributedDataParallel): DDP-wrapped
                `Agent` instance. Must have 'policy' as a prediction key.
            policy_optimizer (torch.optim.Optimizer): Optimizer for policy_net.
            policy_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]):
                Optional learning rate scheduler for policy_optimizer.
            value_net (Optional[torch.nn.parallel.DistributedDataParallel]):
                Optional DDP-wrapped `Agent` instance. If not None, must have a
                'value_{reward_name}' prediction key for each reward_name in
                env.reward_spec.keys() other than 'extrinsic_raw'.
            value_optimizer (Optional[torch.optim.Optimizer]): Optional
                optimizer for value_net. Required if value_net is not None.
            value_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]):
                Optional learning rate scheduler for value_optimizer.
            standardize_adv (bool): Standardize advantages per trajectory
                segment?
            opt_epochs (int): Optimization epochs per policy improvement phase
                in PPO.
            learner_batch_size (int): Batch size per learner process.
            clip_param_init (float): Initial PPO clip parameter.
            clip_param_final (float): Final PPO clip parameter, to which
                clip_param_init will be linearly annealed.
            ent_coef_init (float): Initial entropy bonus coefficient.
            ent_coef_final (float): Final entropy bonus coefficient, to which
                ent_coef_init will be linearly annealed.
            vf_loss_cls (str): Value function loss class name. Name must match
                a derived class of _Loss in torch.nn.modules.loss.
                The most useful classes are MSELoss and SmoothL1Loss.
            vf_loss_coef (float): Value function loss coefficient.
                Ignored if value network is separate from policy network.
            vf_loss_clipping (bool): If true, use pessimistic value function
                loss.
            vf_simple_weighting (bool): If true, use equal weighting of all
                value function losses. Ignored if env.reward_spec.keys() does
                not contain any intrinsic rewards.
            use_pcgrad (bool): Use the PCGrad algorithm from Yu et al., 2020?
                Only allowed if policy and value architecture is shared.
            reward_weights (Optional[Mapping[str, float]): Optional reward
                weights mapping, keyed by reward name. Ignored if
                env.reward_spec.keys() does not contain any intrinsic rewards.
                Required if it does. Default: None.
        """
        super().__init__(
            rank=rank,
            world_size=world_size,
            rollout_len=rollout_len,
            extra_steps=extra_steps,
            credit_assignment_ops=credit_assignment_ops,
            stats_window_len=stats_window_len,
            non_learning_steps=non_learning_steps,
            max_steps=max_steps,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            media_dir=media_dir,
            global_step=global_step,
            env=env,
            policy_net=policy_net,
            policy_optimizer=policy_optimizer,
            policy_scheduler=policy_scheduler,
            value_net=value_net,
            value_optimizer=value_optimizer,
            value_scheduler=value_scheduler,
            standardize_adv=standardize_adv,
            reward_weights=reward_weights)

        self._opt_epochs = opt_epochs
        self._learner_batch_size = learner_batch_size
        self._clip_param = LinearSchedule(
            clip_param_init, clip_param_final, max_steps)
        self._ent_coef = LinearSchedule(
            ent_coef_init, ent_coef_final, max_steps)
        self._vf_loss_criterion = get_loss(vf_loss_cls)
        self._vf_loss_coef = vf_loss_coef
        self._vf_loss_clipping = vf_loss_clipping
        self._vf_simple_weighting = vf_simple_weighting
        self._use_pcgrad = use_pcgrad

        self._sampler = IndependentSampler(
            rollout_len=rollout_len, batch_size=learner_batch_size)

    def compute_losses_and_metrics(
            self, minibatch: Dict[str, NestedTensor],
            no_grad: bool) -> Dict[str, tc.Tensor]:
        with tc.no_grad() if no_grad else ExitStack():
            minibatch_new = self.annotate(rollout=minibatch, no_grad=no_grad)
            entropy_dict = ppo_policy_entropy_bonus(
                entropies=minibatch_new['entropies'],
                ent_coef=self._ent_coef.value(self._global_step))
            policy_dict = ppo_policy_surrogate_objective(
                logprobs_new=minibatch_new['logprobs'],
                logprobs_old=minibatch['logprobs'],
                advantages=minibatch['advantages'],
                clip_param=self._clip_param.value(self._global_step),
                reward_weights=self._get_reward_weights())
            value_dict = ppo_vf_loss(
                vpreds_new=minibatch_new['vpreds'],
                vpreds_old=minibatch['vpreds'],
                vtargs=minibatch['vtargs'],
                clip_param=self._clip_param.value(self._global_step),
                vf_loss_criterion=self._vf_loss_criterion,
                vf_loss_clipping=self._vf_loss_clipping,
                vf_simple_weighting=self._vf_simple_weighting,
                reward_weights=self._get_reward_weights())
            policy_objective = policy_dict['policy_surrogate_objective'] + \
                               entropy_dict['policy_entropy_bonus']
            policy_loss = -policy_objective
            value_loss = value_dict['vf_loss']
            vf_coef = 1 if self._value_net else self._vf_loss_coef
            composite_loss = policy_loss + vf_coef * value_loss
            return {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'composite_loss': composite_loss,
                'meanent': entropy_dict['state_averaged_entropy'],
                'clipfrac': policy_dict['clipfrac']
            }

    def _extract_losses(
        self, losses: Dict[str, tc.Tensor]
    ) -> Tuple[Dict[str, tc.Tensor], Dict[str, tc.Tensor]]:
        policy_losses = dict()
        value_losses = dict()
        if not self._use_pcgrad:
            policy_losses['composite_loss'] = losses['composite_loss']
            value_losses['composite_loss'] = losses['composite_loss']
            return policy_losses, value_losses
        policy_losses['policy_loss'] = losses['policy_loss']
        if self._value_net:
            value_losses['value_loss'] = losses['value_loss']
        else:
            policy_losses['value_loss'] = losses['value_loss']
        return policy_losses, value_losses

    def _optimize_losses(
            self,
            net: DDP,
            optimizer: Optimizer,
            losses: Dict[str, tc.Tensor],
            retain_graph: bool) -> None:
        optimizer.zero_grad()
        if not self._use_pcgrad:
            losses['composite_loss'].backward(retain_graph=retain_graph)
        else:
            apply_pcgrad(
                network=net,
                optimizer=optimizer,
                task_losses=losses,
                normalize=False)
        optimizer.step()

    def optimize(self, rollout: NestedTensor) -> None:
        for opt_epoch in range(self._opt_epochs):
            self._sampler.resample()
            for minibatch_indices in self._sampler:
                minibatch = slice_nested_tensor(rollout, minibatch_indices)
                update_trainable_wrappers(self._env, minibatch)
                if self._global_step <= self._non_learning_steps:
                    continue
                losses = self.compute_losses_and_metrics(
                    minibatch=minibatch, no_grad=False)
                policy_losses, value_losses = self._extract_losses(losses)
                self._optimize_losses(
                    net=self._policy_net,
                    optimizer=self._policy_optimizer,
                    losses=policy_losses,
                    retain_graph=self._value_net is not None)
                if self._value_net:
                    self._optimize_losses(
                        net=self._value_net,
                        optimizer=self._value_optimizer,
                        losses=value_losses,
                        retain_graph=False)

            metrics = self.compute_losses_and_metrics(
                minibatch=rollout, no_grad=True)
            global_metrics = global_means(
                local_values=metrics, world_size=self._world_size, item=True)
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

    def persist(self, metadata: Dict[str, Any]) -> None:
        global_metadata = global_gathers(
            local_lists=metadata, world_size=self._world_size)
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
                        **self._env.checkpointables
                    },
                    steps=self._global_step)


def ppo_policy_entropy_bonus(entropies: tc.Tensor,
                             ent_coef: float) -> Dict[str, tc.Tensor]:
    """
    Computes state-averaged policy entropy.

    Args:
        entropies (torch.Tensor): Policy entropy at each visited state.
        ent_coef (float): PPO entropy bonus coefficient.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of mapping from names to losses
            and metrics.
    """
    meanent = tc.mean(entropies)
    policy_entropy_bonus = ent_coef * meanent
    return {
        'state_averaged_entropy': meanent,
        'policy_entropy_bonus': policy_entropy_bonus
    }


def ppo_policy_surrogate_objective(
        logprobs_new: tc.Tensor,
        logprobs_old: tc.Tensor,
        advantages: Mapping[str, tc.Tensor],
        clip_param: float,
        reward_weights: Mapping[str, float]) -> Dict[str, tc.Tensor]:
    """
    Computes PPO policy surrogate objective.

    Args:
        logprobs_new (torch.Tensor): New log probabilities for actions taken.
        logprobs_old (torch.Tensor): Old log probabilities for actions taken.
        advantages (Mapping[str, torch.Tensor]): Dictionary mapping from reward
            names to advantage estimates.
        clip_param (float): PPO clip parameter epsilon.
        reward_weights (Mapping[str, float]): Dictionary mapping from reward
            names to reward weights.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of mapping from names to losses
            and metrics.
    """
    policy_ratios = tc.exp(logprobs_new - logprobs_old)
    clipped_policy_ratios = tc.clip(
        policy_ratios, 1. - clip_param, 1. + clip_param)
    advantage_shape = advantages['extrinsic'].shape
    advantage_acc = tc.zeros(advantage_shape, dtype=tc.float32)
    for key in reward_weights:
        advantage_acc += reward_weights[key] * advantages[key]
    surr1 = advantage_acc * policy_ratios
    surr2 = advantage_acc * clipped_policy_ratios
    policy_surrogate_objective = tc.mean(tc.min(surr1, surr2))
    clipfrac = tc.mean(tc.greater(surr1, surr2).float())
    return {
        'policy_surrogate_objective': policy_surrogate_objective,
        'clipfrac': clipfrac
    }


def ppo_vf_loss(
        vpreds_new: Mapping[str, tc.Tensor],
        vpreds_old: Mapping[str, tc.Tensor],
        vtargs: Mapping[str, tc.Tensor],
        clip_param: float,
        vf_loss_criterion: tc.nn.modules.loss._Loss,
        vf_loss_clipping: bool,
        vf_simple_weighting: bool,
        reward_weights: Mapping[str, float]) -> Dict[str, tc.Tensor]:
    """
    Computes PPO value surrogate objective.

    Args:
        vpreds_new (Mapping[str, torch.Tensor]): Dictionary mapping from
            reward names to new value predictions.
        vpreds_old (Mapping[str, torch.Tensor]): Dictionary mapping from
            reward names to old value predictions.
        vtargs (Mapping[str, torch.Tensor]): Dictionary mapping from
             reward names to value targets.
        clip_param (float): PPO clip parameter epsilon.
        vf_loss_criterion (torch.nn.modules.loss._Loss): Value loss criterion.
            We expect that the reduction is 'none'.
        vf_loss_clipping (bool): Use pessimistic PPO value loss?
        vf_simple_weighting (bool): Equal weight on each reward's value loss?
        reward_weights (Mapping[str, float]):  Dictionary mapping from reward
            names to reward weights.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of mapping from names to losses
            and metrics.
    """
    vf_loss = tc.tensor(0.0)
    for k in reward_weights:
        if vf_loss_clipping:
            vpreds_new_clipped_k = vpreds_old[k] + tc.clip(
                vpreds_new[k] - vpreds_old[k], -clip_param, clip_param)
            vsurr1 = vf_loss_criterion(input=vpreds_new[k], target=vtargs[k])
            vsurr2 = vf_loss_criterion(
                input=vpreds_new_clipped_k, target=vtargs[k])
            vf_loss_for_rew = tc.mean(tc.max(vsurr1, vsurr2))
        else:
            vf_loss_for_rew = tc.mean(
                vf_loss_criterion(input=vpreds_new[k], target=vtargs[k]))
        if vf_simple_weighting:
            vf_loss += vf_loss_for_rew
        else:
            vf_loss += (reward_weights[k]**2) * vf_loss_for_rew
    return {'vf_loss': vf_loss}
