from drl.agents.preprocessing.abstract import Preprocessing


class EndToEndPreprocessing(Preprocessing):
    def __init__(self, preprocessing_stack):
        super().__init__()
        self._preprocessing_stack = preprocessing_stack

    def forward(self, x):
        return self._preprocessing_stack(x)
