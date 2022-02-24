"""
Normalization utility.
"""

from typing import List, Optional

import torch as tc


class Normalizer(tc.nn.Module):
    """
    Normalizer that maintains empirical estimates of the first and second
    noncentral moments of a stream of data.
    """
    def __init__(
            self,
            data_shape: List[int],
            clip_low: Optional[float] = None,
            clip_high: Optional[float] = None):
        """
        Args:
            data_shape (List[int]): Input data shape.
            clip_low (Optional[float]): Optional lower bound for clipping the
                data after normalization.
            clip_high (Optional[float]): Optional upper bound for clipping the
                data after normalization.
        """
        super().__init__()
        self._data_shape = data_shape
        self._clip_low = clip_low
        self._clip_high = clip_high
        self.register_buffer("_steps", tc.tensor(0.))
        self.register_buffer("_moment1", tc.zeros(data_shape, dtype=tc.float32))
        self.register_buffer("_moment2", tc.zeros(data_shape, dtype=tc.float32))

    @property
    def steps(self) -> tc.Tensor:
        """
        Steps getter.

        Returns:
            torch.Tensor: steps.
        """
        return self._steps

    @steps.setter
    def steps(self, value: tc.Tensor) -> None:
        """
        Steps setter.

        Args:
            value (torch.Tensor): Value for steps.

        Returns:
            None.
        """
        self.register_buffer('_steps', value)

    @property
    def moment1(self) -> tc.Tensor:
        """
        First noncentral moment getter.

        Returns:
            torch.Tensor: Element-wise first noncentral moment estimate.
        """
        return self._moment1

    @moment1.setter
    def moment1(self, value: tc.Tensor) -> None:
        """
        First noncentral moment getter.

        Args:
            value (torch.Tensor): Element-wise first noncentral moment estimates.

        Returns:
            None.
        """
        self.register_buffer('_moment1', value)

    @property
    def moment2(self):
        """
        Second noncentral moment getter.

        Returns:
            torch.Tensor: Element-wise second noncentral moment estimate.
        """
        return self._moment2

    @moment2.setter
    def moment2(self, value):
        """
        Second noncentral moment getter.

        Args:
            value (torch.Tensor): Element-wise second noncentral moment estimates.

        Returns:
            None.
        """
        self.register_buffer('_moment2', value)

    def update(self, item: tc.Tensor) -> None:
        """
        Update normalizer with a new item.

        Args:
            item (torch.Tensor): Torch tensor.

        Returns:
            None.
        """
        steps = self.steps + 1

        moment1 = self.moment1
        moment1 *= ((steps - 1) / steps)
        moment1 += (1 / steps) * item

        moment2 = self.moment2
        moment2 *= ((steps - 1) / steps)
        moment2 += (1 / steps) * tc.square(item)

        self.steps = steps
        self.moment1 = moment1
        self.moment2 = moment2

    def forward(
            self,
            item: tc.Tensor,
            shift: bool = True,
            scale: bool = True,
            eps: float = 1e-8) -> tc.Tensor:
        """
        Applies normalizer to an item without updating internal statistics.

        Args:
            item (torch.tensor): Item to normalize.
            shift (bool): Shift the item by subtracting the mean? Default: True.
            scale (bool): Scale the item by dividing by the standard deviation?
                Default: True.
            eps (float): Numerical stability constant for standard deviation.
                Default: 1e-4.

        Returns:
            torch.Tensor: Normalized and optionally clipped item.
        """
        m1, m2 = self.moment1.unsqueeze(0), self.moment2.unsqueeze(0)
        mean = m1
        var = m2 - tc.square(m1)
        if shift:
            item -= mean
        if scale:
            item *= tc.rsqrt(var + eps)
        if self._clip_low is not None:
            lows = tc.ones_like(item) * self._clip_low
            item = tc.max(lows, item)
        if self._clip_high is not None:
            highs = tc.ones_like(item) * self._clip_high
            item = tc.min(item, highs)
        return item
