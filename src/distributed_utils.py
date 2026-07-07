import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int]:
    """
    torchrun が設定する環境変数 (RANK, LOCAL_RANK, WORLD_SIZE) から
    プロセスグループを初期化する。torchrun 経由で起動されていない場合は
    何もせず (rank=0, local_rank=0) を返す。

    Returns:
        (rank, local_rank)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """tensor を in-place で全プロセス合計する。非分散環境では何もしない。"""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
