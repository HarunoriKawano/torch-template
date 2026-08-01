from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING

import torch
from torch import nn

from configs import ModelParameters

if TYPE_CHECKING:
    # BatchState/FitContextはメソッドの型ヒントとしてのみ使われるため、
    # 実行時のimportは不要（configs->task_module->states->...の循環importを避けるため遅延させている）
    from states import BatchState, FitContext

# TODO modelをラップするモジュールを定義し、lossの計算や予測値の出力までを完結させる
class TaskModule(nn.Module):
    def __init__(self, model_parameters: ModelParameters):
        super().__init__()
        self.model_parameters = model_parameters


    def forward(
        self, batch_state: BatchState, fit_context: Optional[FitContext],
        mode: Literal["train", "val", "test"]
    ) -> BatchState:
        # Engineはこのモジュール自体をDistributedDataParallelでラップするため、
        # train_step/val_step/test_stepを直接呼ばず必ずこのforward経由で呼び出す
        # (DDPはforward()の呼び出しを経由して初めて勾配同期を正しく準備するため)。
        # train/val/testで出力が変わりうるため、self.training(2値)ではなくmodeで明示的に分岐する。
        if mode == "train" and fit_context is not None:
            return self.train_step(batch_state, fit_context)
        if mode == "val":
            return self.val_step(batch_state)
        if mode == "test":
            return self.test_step(batch_state)
        raise ValueError(f"unknown mode: {mode}")

    def train_step(self, batch_state: BatchState, fit_context: FitContext) -> BatchState:
        # 学習のstepを行い、batch stateを更新して返す
        ...

    def val_step(self, batch_state: BatchState) -> BatchState:
        ...

    def test_step(self, batch_state: BatchState) -> BatchState:
        return self.val_step(batch_state)

    def predict(self, *args, **kwargs) -> Any:...

    def save(self, save_dir: str) -> None: ...

    def load(self, save_dir: str, device: str = "cpu") -> None: ...
