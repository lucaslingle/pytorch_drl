from typing import Union

import abc

import torch as tc


class Head(tc.nn.Module, metaclass=abc.ABCMeta):
    """
    Runs a prediction op.
    """
    @abc.abstractmethod
    def forward(
            self, features: tc.Tensor
    ) -> Union[tc.Tensor, tc.distributions.Distribution]:
        """
        Forward method.

        Args:
            features (torch.Tensor): Torch tensor with two dimensions.

        Returns:
            Union[torch.Tensor, torch.distributions.Distribution]: Predictions.
        """
