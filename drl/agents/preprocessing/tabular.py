import torch as tc

from drl.agents.preprocessing import Preprocessing


def one_hot(ys: tc.Tensor, depth: int) -> tc.Tensor:
    """
    Applies one-hot encoding to a batch of vectors.

    Args:
        ys (torch.Tensor): Torch tensor with shape batch_shape, and dtype int64.
        depth (int): The number of possible categorical values.

    Returns:
        torch.Tensor: Torch tensor of shape [*batch_shape, depth] and dtype float32
        containing one-hot encodings for each batch element.
    """
    vecs_shape = list(ys.shape) + [depth]
    vecs = tc.zeros(dtype=tc.float32, size=vecs_shape)
    vecs.scatter_(
        dim=-1,
        index=ys.unsqueeze(-1),
        src=tc.ones(dtype=tc.float32, size=vecs_shape))
    return vecs.float()


class OneHotEncode(Preprocessing):
    def __init__(self, depth: int):
        """
        Args:
            depth (int): Number of possible categorical values.
        """
        super().__init__()
        self._depth = depth

    def forward(self, x, **kwargs):
        return one_hot(x, depth=self._depth)
