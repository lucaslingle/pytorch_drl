import abc

import torch as tc


class Architecture(tc.nn.Module, metaclass=abc.ABCMeta):
    """
    Computes features from an input.
    """
