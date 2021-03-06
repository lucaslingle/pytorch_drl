from drl.agents.preprocessing.abstract import Preprocessing


class ToChannelMajor(Preprocessing):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x.permute(0, 3, 1, 2)
