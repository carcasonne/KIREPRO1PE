import random
from typing import Generic, Tuple, TypeVar, Union

from torch.utils.data import DataLoader

T = TypeVar("T")
L = TypeVar("L")


class AudioDataLoader(DataLoader, Generic[T, L]):
    def get_sample(self, get_label: bool = False) -> Union[T, Tuple[T, L]]:
        """yoink a random sample from the dataloader"""
        batch = next(iter(self))
        specs, labels = batch
        idx = random.randint(0, len(specs) - 1)
        return (specs[idx], labels[idx]) if get_label else specs[idx]
