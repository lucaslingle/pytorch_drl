from drl.agents.architectures.stateless.abstract import (
    StatelessArchitecture, HeadEligibleArchitecture)
from drl.agents.architectures.stateless.async_cnn import AsyncCNN
from drl.agents.architectures.stateless.dueling import DuelingArchitecture
from drl.agents.architectures.stateless.identity import Identity
from drl.agents.architectures.stateless.linear import Linear
from drl.agents.architectures.stateless.mlp import MLP
from drl.agents.architectures.stateless.nature_cnn import NatureCNN

__all__ = [
    "StatelessArchitecture",
    "HeadEligibleArchitecture",
    "AsyncCNN",
    "DuelingArchitecture",
    "Identity",
    "Linear",
    "MLP",
    "NatureCNN"
]
