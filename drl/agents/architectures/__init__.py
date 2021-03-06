from drl.agents.architectures.abstract import Architecture
from drl.agents.architectures.stateless import (
    StatelessArchitecture,
    HeadEligibleArchitecture,
    AsyncCNN,
    DuelingArchitecture,
    Identity,
    Linear,
    MLP,
    NatureCNN)
from drl.agents.architectures.stateful import StatefulArchitecture

__all__ = [
    "Architecture",
    "StatelessArchitecture",
    "HeadEligibleArchitecture",
    "AsyncCNN",
    "DuelingArchitecture",
    "Identity",
    "Linear",
    "MLP",
    "NatureCNN",
    "StatefulArchitecture"
]
