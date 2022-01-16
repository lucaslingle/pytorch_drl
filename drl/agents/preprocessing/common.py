from drl.agents.preprocessing.abstract import Preprocessing


class EndToEndPreprocessing(Preprocessing):
    def __init__(self, preprocessing_stack, detach_input=True):
        super().__init__()
        self._detach_input = detach_input
        self._preprocessing_stack = preprocessing_stack

    def forward(self, x):
        if self._detach_input:
            x = x.detach()
        return self._preprocessing_stack(x)
