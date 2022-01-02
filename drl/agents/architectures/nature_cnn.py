import torch as tc

from drl.agents.architectures.abstract import Architecture


class NatureCNN(Architecture):
    """
    Implements the convolutional torso of the agent from Mnih et al., 2015
    - 'Human Level Control through Deep Reinforcement Learning'.
    """
    def __init__(self, preprocessing, img_channels):
        super().__init__(preprocessing)
        self._num_features = 512
        self._network = tc.nn.Sequential(
            tc.nn.Conv2d(img_channels, 32, kernel_size=(8,8), stride=(4,4)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1)),
            tc.nn.ReLU(),
            tc.nn.Flatten(),
            tc.nn.Linear(3136, self._num_features),
            tc.nn.ReLU()
        )

    @property
    def output_dim(self):
        return self._num_features

    def features(self, x):
        features = self._network(self._preprocessing(x))
        return features
