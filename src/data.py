from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from configs import HyperParameters
from states import BatchState, SizedIterable
from distributed_utils import is_distributed

# TODO Implement the original dataset
class CustomizedDataset(Dataset):
    def __len__(self) -> int:...

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:...

# TODO Data loaderの返り値がbatch stateになるように更新する
def dataset_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> BatchState:...
    # targets, labels = list(zip(*batch))

def get_dataloader(
    dataset: Dataset, hyper_parameters: HyperParameters, shuffle: bool, cpu_num_works: int = 4
) -> DataLoader[BatchState]:
    sampler: Optional[DistributedSampler] = None
    if is_distributed():
        # DDP環境ではプロセス毎にデータを分割するDistributedSamplerを使う。
        # sampler使用時はDataLoaderのshuffle引数と併用できないため、shuffleはsampler側に渡す。
        # drop_lastでプロセス間のバッチ数を揃え、collective通信の不整合(ハング)を防ぐ。
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        shuffle = False

    dataloader = DataLoader(
        dataset, shuffle=shuffle, sampler=sampler, batch_size=hyper_parameters.batch_size,
        num_workers=cpu_num_works,
        pin_memory=True, persistent_workers=True, collate_fn=dataset_collate_fn
    )

    return dataloader


def set_dataloader_epoch(dataloader: SizedIterable, epoch: int) -> None:
    """DistributedSamplerを使っている場合、epoch毎にset_epochを呼びシャッフル系列を切り替える"""
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


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
