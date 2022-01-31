from drl.agents.architectures.stateless.async_cnn import AsyncCNN
from drl.agents.architectures.stateless.dueling import DuelingArchitecture
from drl.agents.architectures.stateless.linear import Linear
from drl.agents.architectures.stateless.mlp import MLP

__all__ = [
    "StatelessArchitecture",
    "HeadEligibleArchitecture",
    "AsyncCNN",
    "DuelingArchitecture",
    "Linear",
    "MLP",
    "NatureCNN"
]
