from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar

import numpy as np

from DLpy.core.tensor import Tensor

from .dataset import Dataset
from .samplers import RandomSampler, Sampler, SequentialSampler

# Define a type variable for our batch elements
T = TypeVar("T")


class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The DataLoader coordinates how data is loaded and batched during training:
    - It uses a sampler to determine the order of data loading
    - It collates individual samples into batches
    - It optionally shuffles the data at each epoch

    Args:
        dataset: Dataset from which to load the data
        batch_size: How many samples per batch to load (default: 1)
        shuffle: Set to True to have the data reshuffled at every epoch
        sampler: Defines the strategy to draw samples from the dataset
        drop_last: If True, drop the last incomplete batch
        collate_fn: Merges a list of samples to form a mini-batch
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,  # Changed from Optional[int] to int with default
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable[[Sequence[T]], Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size  # Now guaranteed to be an int
        self.drop_last = drop_last
        # Store collate function, using default if none provided
        self.collate_fn = collate_fn if collate_fn is not None else self._default_collate

        # Set up the sampler based on input parameters
        if sampler is not None:
            if shuffle:
                raise ValueError("Cannot specify both shuffle and sampler")
            self.sampler = sampler
        else:
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)

    def _default_collate(self, batch: Sequence[Any]) -> Any:
        """
        Default collate function for batching samples.

        This method handles various types of data:
        - Tensors: Stacks them along a new first dimension
        - Scalars: Converts them to a single tensor
        - Tuples/Lists: Recursively collates their elements
        - Other types: Attempts to convert to tensor or returns as-is

        Args:
            batch: A sequence of samples to be collated

        Returns:
            The collated batch in an appropriate format
        """
        elem = batch[0]

        if isinstance(elem, Tensor):
            return Tensor(np.stack([b.data for b in batch]))
        elif isinstance(elem, (int, float)):
            # Convert scalar values to a flat tensor
            return Tensor(np.array(batch))
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return type(elem)(*(self._default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, (list, tuple)):
            return [self._default_collate(samples) for samples in zip(*batch)]
        else:
            try:
                return Tensor(np.array(batch))
            except:
                return batch

    def __iter__(self) -> Iterator[Any]:
        """
        Creates an iterator over the dataset, yielding batches of data.

        The iterator:
        1. Uses the sampler to determine the order of samples
        2. Collects samples into batches of size batch_size
        3. Applies the collate function to create the final batch

        Returns:
            An iterator yielding batched data
        """
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        # Handle the last batch if it exists and we're not dropping it
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        """
        Calculates the number of batches in the dataset.

        Takes into account:
        - The total dataset size
        - The batch size
        - Whether we're dropping the last incomplete batch

        Returns:
            The number of batches that will be yielded
        """
        # Since batch_size is now a regular int, we can do arithmetic safely
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
