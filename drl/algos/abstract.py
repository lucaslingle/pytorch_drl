from typing import Dict, Union
import abc

import torch as tc

from drl.utils.types import NestedTensor


class Algo(metaclass=abc.ABCMeta):
    # todo: add args to parent class including
    #     world_size, seg_len, extra_steps,
    #     credit_assignment_spec,
    #     policy_net,
    #     stats_window_len,
    #     log_dir,
    #     media_dir.
    #     then instantate trajectory mgr, tensorboard summary writer, etc. here.
    #     and make a generic video-saving method here.
    def __init__(self, rank):
        self._rank = rank

    @abc.abstractmethod
    def annotate(self, trajectory: Dict[str, NestedTensor],
                 no_grad: bool) -> Dict[str, NestedTensor]:
        """
        Forward pass through the networks.
        """
        pass

    @abc.abstractmethod
    def credit_assignment(
            self, trajectory: Dict[str,
                                   NestedTensor]) -> Dict[str, NestedTensor]:
        """
        Assign credit backwards in time.
        """
        pass

    @abc.abstractmethod
    def compute_losses(self, mb: Dict[str, NestedTensor],
                       no_grad: bool) -> Dict[str, tc.Tensor]:
        """
        Compute losses for learning.
        """
        pass

    @abc.abstractmethod
    def training_loop(self) -> None:
        """
        Training loop.
        """
        pass

    @abc.abstractmethod
    def evaluation_loop(self) -> Dict[str, Union[float, tc.Tensor]]:
        """
        Evaluation loop.
        """
        pass
