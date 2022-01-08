import torch as tc

from drl.agents.architectures.abstract import Architecture


class AsyncCNN(Architecture):
    """
    Implements the convolutional torso of the agent from Mnih et al., 2016
    - 'Asynchronous Methods for Deep Reinforcement Learning'.
    """
    def __init__(self, img_channels):
        super().__init__()
        self._num_features = 256
        self._network = tc.nn.Sequential(
            tc.nn.Conv2d(img_channels, 16, kernel_size=(8,8), stride=(4,4)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(16, 32, kernel_size=(4,4), stride=(2,2)),
            tc.nn.ReLU(),
            tc.nn.Flatten(),
            tc.nn.Linear(2592, self._num_features),
            tc.nn.ReLU()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._network:
            if isinstance(m, (tc.nn.Linear, tc.nn.Conv2d)):
                tc.nn.init.orthogonal_(m.weight, gain=1.0)
                tc.nn.init.zeros_(m.bias)

    @property
    def output_dim(self):
        return self._num_features

    def forward(self, x, **kwargs):
        features = self._network(x)
        return features
