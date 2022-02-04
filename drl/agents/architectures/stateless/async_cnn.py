from typing import Callable, List

import torch as tc

from drl.agents.architectures.stateless.abstract import StatelessArchitecture


class AsyncCNN(StatelessArchitecture):
    """
    Reference: Mnih et al., 2016 -
        'Asynchronous Methods for Deep Reinforcement Learning'.
    """
    def __init__(
        self,
        img_channels: int,
        w_init: Callable[[tc.Tensor], None],
        b_init: Callable[[tc.Tensor], None],
    ):
        """
        Args:
            img_channels (int): Image channels.
            w_init (Callable[[torch.Tensor], None]): Weight initializer.
            b_init (Callable[[torch.Tensor], None]): Bias initializer.
        """
        super().__init__(w_init, b_init)
        self._img_channels = img_channels
        self._num_features = 256
        self._network = tc.nn.Sequential(
            tc.nn.Conv2d(img_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            tc.nn.ReLU(),
            tc.nn.Flatten(),
            tc.nn.Linear(2592, self._num_features),
            tc.nn.ReLU())
        self._init_weights(self._network)

    @property
    def input_shape(self) -> List[int]:
        shape = [self._img_channels, 84, 84]
        return shape

    @property
    def output_dim(self) -> int:
        return self._num_features

    def forward(self, x, **kwargs):
        features = self._network(x)
        return features
