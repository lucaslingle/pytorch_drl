from typing import Iterator
import abc

import numpy as np

from drl.utils.types import Indices


class Sampler(metaclass=abc.ABCMeta):
    def __init__(self, rollout_len: int) -> None:
        self._rollout_len = rollout_len

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abc.abstractmethod
    def resample(self) -> None:
        """
        Reshuffle indices.
        """
        pass


class IndicesIterator(object):
    def __init__(self, indices: Indices, stepsize: int) -> None:
        assert len(indices.shape) == 1
        assert indices.shape[0] % stepsize == 0
        self._indices = indices
        self._stepsize = stepsize
        self._offset = 0

    def __iter__(self) -> 'IndicesIterator':
        return self

    def __next__(self) -> Indices:
        start_idx = self._offset
        end_idx = self._offset + self._stepsize
        try:
            next_indices = self._indices[start_idx:end_idx]
            self._offset += self._stepsize
            if next_indices.size == 0:
                raise StopIteration
            return next_indices
        except IndexError:
            raise StopIteration


class IndependentSampler(Sampler):
    def __init__(self, rollout_len: int, batch_size: int) -> None:
        assert rollout_len % batch_size == 0
        super().__init__(rollout_len)
        self._batch_size = batch_size
        self._iterator = None
        self.resample()

    def __iter__(self):
        return self._iterator

    def resample(self):
        self._iterator = IndicesIterator(
            indices=np.random.permutation(self._rollout_len),
            stepsize=self._batch_size)


class TBPTTSampler(Sampler):
    def __init__(self, rollout_len: int, tbptt_len: int) -> None:
        assert rollout_len % tbptt_len == 0
        super().__init__(rollout_len)
        self._tbptt_len = tbptt_len
        self._iterator = None
        self.resample()

    def __iter__(self):
        return self._iterator

    def resample(self):
        consecutive = np.arange(0, self._rollout_len)
        consecutive_2d = consecutive.reshape((-1, self._tbptt_len))
        row_perm = np.random.permutation(consecutive_2d.shape[0])
        self._iterator = IndicesIterator(
            indices=consecutive_2d[row_perm].reshape((self._rollout_len, )),
            stepsize=self._tbptt_len)
