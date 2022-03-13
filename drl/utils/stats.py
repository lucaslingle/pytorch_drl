import torch as tc


def standardize(tensor: tc.Tensor) -> tc.Tensor:
    """
    Takes a tensor returns another tensor whose elements are the empirical
    z-scores of the provided tensor.

    Args:
        tensor (torch.Tensor): Torch tensor to standardize.

    Returns:
        torch.Tensor: Torch tensor with standardized entries.
    """
    centered = tensor - tc.mean(tensor)
    scaled = centered / tc.std(tensor)
    return scaled
