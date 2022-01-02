import torch as tc

from drl.agents.preprocessing.abstract import Preprocessing


class ToChannelMajor(Preprocessing):
    def __init__(self):
        super().__init__()

    def preprocessed(self, x):
        return x.permute(0, 3, 1, 2)
