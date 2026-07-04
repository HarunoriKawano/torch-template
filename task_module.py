from typing import Any

import torch
from torch import nn

from states import BatchState, FitContext

# TODO modelをラップするモジュールを定義し、lossの計算や予測値の出力までを完結させる
class TaskModule(nn.Module):

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
