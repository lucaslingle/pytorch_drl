from typing import Mapping, Dict, Union
import abc
import collections
import uuid
import os

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import moviepy.editor as mpy

from drl.algos.common import TrajectoryManager, MultiQueue
from drl.envs.wrappers import Wrapper
from drl.algos.common.credit_assignment import CreditAssignmentOp
from drl.utils.types import NestedTensor


class Algo(metaclass=abc.ABCMeta):
    def __init__(
            self,
            rank: int,
            world_size: int,
            seg_len: int,
            extra_steps: int,
            credit_assignment_ops: Mapping[str, CreditAssignmentOp],
            env: Union[gym.core.Env, Wrapper],
            rollout_net: DDP,
            stats_window_len: int,
            log_dir: str,
            media_dir: str) -> None:
        self._rank = rank
        self._world_size = world_size
        self._seg_len = seg_len
        self._extra_steps = extra_steps
        self._credit_assignment_ops = credit_assignment_ops
        self._env = env
        self._rollout_net = rollout_net
        self._stats_world_len = stats_window_len
        self._log_dir = log_dir
        self._media_dir = media_dir

        self._trajectory_mgr = TrajectoryManager(
            env=env,
            rollout_net=rollout_net,
            seg_len=seg_len,
            extra_steps=extra_steps)
        self._metadata_acc = MultiQueue(maxlen=stats_window_len)
        if self._rank == 0:
            self._writer = SummaryWriter(log_dir)

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
    def training_loop(self) -> None:
        """
        Training loop.

        Returns:
            None.
        """

    @abc.abstractmethod
    def evaluation_loop(self) -> Dict[str, Union[float, tc.Tensor]]:
        """
        Evaluation loop.

        Returns:
            Dict[str, Union[float, tc.Tensor]]: Dictionary mapping from names
            to metrics.
        """

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
                predictions_t = self._rollout_net(
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
