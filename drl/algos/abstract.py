from typing import Mapping, Dict, Union, Optional, Tuple, List
from contextlib import ExitStack
import collections
import uuid
import os

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import moviepy.editor as mpy

from drl.envs.wrappers import Wrapper
from drl.algos.common import (
    TrajectoryManager,
    MultiQueue,
    CreditAssignmentOp,
    AdvantageEstimator,
    extract_reward_name)
from drl.utils.nested import slice_nested_tensor
from drl.utils.stats import standardize
from drl.utils.types import NestedTensor


class Algo(object):
    """
    Base class for RL algorithms.
    """
    def __init__(
            self,
            rank: int,
            world_size: int,
            seg_len: int,
            extra_steps: int,
            credit_assignment_ops: Mapping[str, CreditAssignmentOp],
            env: Union[gym.core.Env, Wrapper],
            policy_net: DDP,
            stats_window_len: int,
            log_dir: str,
            media_dir: str) -> None:
        """
        Args:
            rank (int): Process rank.
            world_size (int): Total number of processes.
            seg_len (int): Trajectory segment length.
            extra_steps (int): Extra steps required for credit assignment.
                Should be set to n-1 if using n-step return-based advantage
                estimation.
            credit_assignment_ops (Mapping[str, AdvantageEstimator]):
                Mapping from reward names to AdvantageEstimator instances.
            env (Union[gym.core.Env, Wrapper]): Environment instance or wrapped
                environment.
            policy_net (torch.nn.parallel.DistributedDataParallel): DDP-wrapped
                `Agent` instance. Must have 'policy' as a prediction key.
            stats_window_len (int): Window size for moving average of episode
                 metadata.
            log_dir (str): Tensorboard logs directory.
            media_dir (str): Media directory.
        """
        self._rank = rank
        self._world_size = world_size
        self._seg_len = seg_len
        self._extra_steps = extra_steps
        self._credit_assignment_ops = credit_assignment_ops
        self._env = env
        self._policy_net = policy_net
        self._stats_world_len = stats_window_len
        self._log_dir = log_dir
        self._media_dir = media_dir

        self._trajectory_mgr = TrajectoryManager(
            env=env,
            rollout_net=policy_net,
            seg_len=seg_len,
            extra_steps=extra_steps)
        self._metadata_acc = MultiQueue(maxlen=stats_window_len)
        if self._rank == 0:
            self._writer = SummaryWriter(log_dir)

    def _get_reward_keys(self, omit_raw: bool = True) -> List[str]:
        reward_spec = self._env.reward_spec
        assert reward_spec is not None
        reward_keys = reward_spec.keys
        if omit_raw:
            reward_keys = [k for k in reward_keys if not k.endswith('_raw')]
            assert len(reward_keys) > 0
        return reward_keys

    def _get_prediction_keys(self) -> Tuple[List[str], List[str]]:
        reward_keys = self._get_reward_keys(omit_raw=True)
        policy_predict = ['policy']
        value_predict = [f'value_{k}' for k in reward_keys]
        return policy_predict, value_predict

    def annotate(self, trajectory: Dict[str, NestedTensor],
                 no_grad: bool) -> Dict[str, NestedTensor]:
        """
        Annotates trajectory with predictions.

        Args:
            trajectory (Dict[str, NestedTensor]): Trajectory segment.
            no_grad (bool): Disable gradient tape recording?

        Returns:
            Dict[str, NestedTensor]: Prediction-annotated trajectory segment.
        """
        raise NotImplementedError

    def credit_assignment(
            self, trajectory: Dict[str,
                                   NestedTensor]) -> Dict[str, NestedTensor]:
        """
        Assigns credit backwards in time.

        Args:
            trajectory (Dict[str, NestedTensor]): Prediction-annotated
                 trajectory segment.

        Returns:
            Dict[str, NestedTensor]: Prediction-annotated
                 trajectory segment with results of credit assignment.
        """
        raise NotImplementedError

    def compute_losses_and_metrics(
            self, minibatch: Dict[str, NestedTensor],
            no_grad: bool) -> Dict[str, tc.Tensor]:
        """
        Computes losses and metrics.

        Args:
            minibatch (Dict[str, NestedTensor]): Minibatch of experience.
            no_grad (bool): Disable gradient tape recording?

        Returns:
            Dict[str, tc.Tensor]: Dictionary mapping from names
            to metrics.
        """
        raise NotImplementedError

    def training_loop(self) -> None:
        """
        Training loop.

        Returns:
            None.
        """
        raise NotImplementedError

    def evaluation_loop(self) -> Dict[str, Union[float, tc.Tensor]]:
        """
        Evaluation loop.

        Returns:
            Dict[str, Union[float, tc.Tensor]]: Dictionary mapping from names
            to metrics.
        """
        raise NotImplementedError

    def video_loop(self) -> None:
        """
        Video saving loop.
        """
        if self._rank == 0:
            max_frames = 2000
            video_frames_per_second = 64
            frame_queue = collections.deque(maxlen=max_frames)
            t = 0
            r_total = 0.
            o_t = self._env.reset()
            done_t = False
            while not done_t:
                predictions_t = self._policy_net(
                    observations=tc.tensor(o_t).float().unsqueeze(0),
                    predict=['policy'])
                pi_dist_t = predictions_t['policy']
                a_t = pi_dist_t.sample().squeeze(0).detach().numpy()
                o_tp1, r_t, done_t, info_t = self._env.step(a_t)
                raw_frame_tp1 = self._env.render(mode='rgb_array')
                frame_queue.append(raw_frame_tp1)
                t += 1
                r_total += r_t['extrinsic_raw']
                o_t = o_tp1

            filename = f"{uuid.uuid4()}.gif"
            fp = os.path.join(self._media_dir, filename)
            max_idx = len(frame_queue) - 1

            def make_frame(t: int) -> np.ndarray:
                # t will range from 0 to (self.max_frames / self.fps).
                frac_done = t / (max_frames // video_frames_per_second)
                idx = int(max_idx * frac_done)
                return frame_queue[idx]

            clip = mpy.VideoClip(
                make_frame, duration=(max_frames // video_frames_per_second))
            clip.write_gif(fp, fps=video_frames_per_second)


class ActorCriticAlgo(Algo):
    """
    Base class for Actor-Critic RL algorithms.
    """
    def __init__(
            self,
            rank: int,
            world_size: int,
            seg_len: int,
            extra_steps: int,
            credit_assignment_ops: Mapping[str, AdvantageEstimator],
            standardize_adv: bool,
            env: Union[gym.core.Env, Wrapper],
            policy_net: DDP,
            value_net: Optional[DDP],
            stats_window_len: int,
            log_dir: str,
            media_dir: str) -> None:
        """
        Args:
            rank (int): Process rank.
            world_size (int): Total number of processes.
            seg_len (int): Trajectory segment length.
            extra_steps (int): Extra steps required for credit assignment.
                Should be set to n-1 if using n-step return-based advantage
                estimation.
            credit_assignment_ops (Mapping[str, AdvantageEstimator]):
                Mapping from reward names to AdvantageEstimator instances.
            standardize_adv (bool): Standardize advantages per trajectory
                segment?
            env (Union[gym.core.Env, Wrapper]): Environment instance or wrapped
                environment.
            policy_net (torch.nn.parallel.DistributedDataParallel): DDP-wrapped
                `Agent` instance. Must have 'policy' as a prediction key.
            value_net (Optional[torch.nn.parallel.DistributedDataParallel]):
                Optional DDP-wrapped `Agent` instance.
            stats_window_len (int): Window size for moving average of episode
                 metadata.
            log_dir (str): Tensorboard logs directory.
            media_dir (str): Media directory.
        """
        super().__init__(
            rank=rank,
            world_size=world_size,
            seg_len=seg_len,
            extra_steps=extra_steps,
            credit_assignment_ops=credit_assignment_ops,
            env=env,
            policy_net=policy_net,
            stats_window_len=stats_window_len,
            log_dir=log_dir,
            media_dir=media_dir)

        self._credit_assignment_ops = credit_assignment_ops
        self._standardize_adv = standardize_adv
        self._value_net = value_net

    def annotate(self, trajectory: Dict[str, NestedTensor],
                 no_grad: bool) -> Dict[str, NestedTensor]:
        """
        Annotates trajectory with predictions.

        Args:
            trajectory (Dict[str, NestedTensor]): Trajectory segment.
            no_grad (bool): Disable gradient tape recording?

        Returns:
            Dict[str, NestedTensor]: Prediction-annotated trajectory segment.
        """
        with tc.no_grad() if no_grad else ExitStack():
            # make policy and value predictions.
            policy_predict, value_predict = self._get_prediction_keys()
            if not self._value_net:
                policy_predict.extend(value_predict)
            predictions = self._policy_net(
                trajectory['observations'], predict=policy_predict)
            pi = predictions.pop('policy')
            if not self._value_net:
                vpreds = predictions
            else:
                vpreds = self._value_net(
                    trajectory['observations'], predict=value_predict)

            # shallow copy of trajectory dict, point to initial/new values.
            trajectory_new = {
                'observations': trajectory['observations'],
                'actions': trajectory['actions'],
                'logprobs': pi.log_prob(trajectory['actions']),
                'entropies': pi.entropy()
            }
            trajectory_new = slice_nested_tensor(
                trajectory_new, slice(0, self._seg_len))
            trajectory_new.update({
                'rewards': trajectory['rewards'],
                'dones': trajectory['dones'],
                'vpreds': {extract_reward_name(k): vpreds[k] for k in vpreds}
            })
            return trajectory_new

    @tc.no_grad()
    def credit_assignment(
            self, trajectory: Dict[str,
                                   NestedTensor]) -> Dict[str, NestedTensor]:
        """
        Computes advantage estimates and state-value targets.

        Args:
            trajectory (Dict[str, NestedTensor]): Prediction-annotated
                 trajectory segment.

        Returns:
            Dict[str, NestedTensor]: Prediction-annotated
                 trajectory segment with results of credit assignment.
        """
        advantages, td_lambda_returns = dict(), dict()
        for k in self._get_reward_keys(omit_raw=True):
            advantages_k = self._credit_assignment_ops[k](
                seg_len=self._seg_len,
                extra_steps=self._extra_steps,
                rewards=trajectory['rewards'][k],
                vpreds=trajectory['vpreds'][k],
                dones=trajectory['dones'])
            vpreds_k = trajectory['vpreds'][k][slice(0, self._seg_len)]
            td_lambda_returns[k] = advantages_k + vpreds_k
            if self._standardize_adv:
                advantages_k = standardize(advantages_k)
            advantages[k] = advantages_k
        trajectory.update({
            'advantages': advantages,
            'td_lambda_returns': td_lambda_returns,
        })
        trajectory = slice_nested_tensor(trajectory, slice(0, self._seg_len))
        return trajectory

    def compute_losses_and_metrics(
            self, minibatch: Dict[str, NestedTensor],
            no_grad: bool) -> Dict[str, tc.Tensor]:
        """
        Computes losses and metrics.

        Args:
            minibatch (Dict[str, NestedTensor]): Minibatch of experience.
            no_grad (bool): Disable gradient tape recording?

        Returns:
            Dict[str, tc.Tensor]: Dictionary mapping from names
            to metrics.
        """
        raise NotImplementedError

    def training_loop(self) -> None:
        """
        Training loop.

        Returns:
            None.
        """
        raise NotImplementedError

    def evaluation_loop(self) -> Dict[str, Union[float, tc.Tensor]]:
        """
        Evaluation loop.

        Returns:
            Dict[str, Union[float, tc.Tensor]]: Dictionary mapping from names
            to metrics.
        """
        raise NotImplementedError
