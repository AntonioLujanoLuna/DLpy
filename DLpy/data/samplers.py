from typing import Iterator, Optional, Sized

import numpy as np


class Sampler:
    """Base class for all Samplers."""

    def __init__(self, data_source: Optional[Sized]):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially."""

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples elements randomly without replacement."""

    def __iter__(self) -> Iterator[int]:
        indices = np.random.permutation(len(self.data_source))
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.data_source)
