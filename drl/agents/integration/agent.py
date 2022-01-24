from typing import Dict, List

import torch as tc

from drl.agents.preprocessing import Preprocessing
from drl.agents.architectures.stateless.abstract import StatelessArchitecture
from drl.agents.heads import (
    Head
)


class Agent(tc.nn.Module):
    def __init__(
            self,
            preprocessing: List[Preprocessing],
            architecture: StatelessArchitecture,
            predictors: Dict[str, Head],
            detach_input: bool = True
    ):
        """
        Args:
            preprocessing: List of `Preprocessing` instances.
            architecture: `StatelessArchitecture` instance.
            predictors: Dictionary of prediction `Head`s, keyed by predictor name.
                There should be only one `PolicyHead` predictor and its name
                should be 'policy'. There can be multiple `ValueHead`/`ActionValueHead`
                predictors. Their names should start with 'value_' or 'action_value_,
                and end with the appropriate reward name.
            detach_input: Detach input or not? Default: True.
                This is a no-op if used with external observation tensors,
                since these are outside the model.parameters() passed to the
                optimizer and moreover default to have requires_grad=False.
        """
        super().__init__()
        self._preprocessing = tc.nn.Sequential(*preprocessing)
        self._architecture = architecture
        self._predictors = tc.nn.ModuleDict(predictors)
        self._detach_input = detach_input
        self._check_predict_keys()

    def _check_predict_keys(self):
        for key in self.keys:
            if key == 'policy':
                continue
            if key.startswith('value_'):
                continue
            if key.startswith('action_value_'):
                continue
            raise ValueError(f"Prediction key {key} not supported.")

    @property
    def keys(self):
        return self._predictors.keys()

    def forward(self, observations, predict, **kwargs):
        """
        Args:
            observations: Batch of observations
            predict: Names of predictors to apply.
            kwargs: Keyword arguments.
        Returns:
            Dictionary of predictions.
        """
        if self._detach_input:
            observations = observations.detach()
        preprocessed = self._preprocessing(observations)
        features = self._architecture(preprocessed, **kwargs)
        predictions = {
            key: self._predictors[key](features, **kwargs) for key in predict
        }
        return predictions
