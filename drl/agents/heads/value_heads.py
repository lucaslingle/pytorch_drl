import torch as tc

from drl.agents.heads.abstract import ValueHeadMixin


class LinearValueHead(ValueHeadMixin):
    def __init__(self, num_features):
        super().__init__()
        self.__value_head = tc.nn.Linear(num_features, 1)
