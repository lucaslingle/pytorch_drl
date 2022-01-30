from drl.agents.architectures.abstract import Architecture
from drl.agents.architectures.stateless import (
    StatelessArchitecture,
    HeadEligibleArchitecture,
    AsyncCNN,
    DuelingArchitecture,
    Linear,
    MLP,
    NatureCNN
)


__all__ = [
    "Architecture",
    "StatelessArchitecture",
    "HeadEligibleArchitecture",
    "AsyncCNN",
    "DuelingArchitecture",
    "Linear",
    "MLP",
    "NatureCNN"
]
