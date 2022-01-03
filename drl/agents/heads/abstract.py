import abc

import torch as tc


class Head(tc.nn.Module, metaclass=abc.ABCMeta):
    """
    Runs a prediction op.
    """
