from typing import Protocol, Iterator, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

from configs import HyperParameters
from states import BatchState

T = TypeVar('T')

class SizedIterable(Protocol[T]):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T]:
        ...

# TODO Implement the original dataset
class CustomizedDataset(Dataset):
    def __len__(self):...

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:...

# TODO Data loaderの返り値がbatch stateになるように更新する
def dataset_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> BatchState:...
    # targets, labels = list(zip(*batch))

def get_dataloader(dataset: Dataset, hyper_parameters: HyperParameters, shuffle: bool) -> DataLoader[BatchState]:
    dataloader = DataLoader(
        dataset, shuffle=shuffle, batch_size=hyper_parameters.batch_size, num_workers=hyper_parameters.cpu_num_works,
        pin_memory=True, persistent_workers=True, collate_fn=dataset_collate_fn
    )

    return dataloader


class ShortDataLoader:
    def __init__(self, dataloader: SizedIterable[BatchState], num_batches: int):
        self.dl = dataloader
        self.num_batches = num_batches

    def __iter__(self):
        for i, batch in enumerate(self.dl):
            if i >= self.num_batches:
                break
            yield batch

    def __len__(self):
        return min(self.num_batches, len(self.dl))
