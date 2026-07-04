from typing import Optional

from pydantic import BaseModel
import torch
from pydantic import computed_field
from torch.optim.lr_scheduler import LRScheduler

from data import SizedIterable
from configs import HyperParameters
from metrics import Metrics

# TODO バッチ内で完結する値を定義する(入力値やラベルなど)
class BatchState(BaseModel):
    loss: Optional[torch.Tensor] = None

    def to(self, device: str): ...
    # TODO データの保存メモリを移動するコードを記述

# TODO best metricをタスクにあわせて変更
class GlobalState(BaseModel):
    best_metric: float
    patience_count: int = 0
    current_epoch: int = 0
    global_step: int = 0

    def one_step(self) -> None:
        self.global_step += 1

    def one_epoch(self) -> None:
        self.current_epoch += 1

    def check_metric(self, metric: Metrics) -> bool:
        better_result = self._metric_update(metric)
        if better_result:
            self.patience_count = 0
            print("\n精度が更新されました。\n")
        else:
            self.patience_count += 1
            print("\n精度は更新されていません。\n")

        return better_result

    def save(self, path: str) -> None:
        json_str = self.model_dump_json()
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    @classmethod
    def load(cls, path: str) -> "GlobalState":
        with open(path, "r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.model_validate_json(json_data)

    def _metric_update(self, metric: Metrics) -> bool: ...


class FitContext(BaseModel):
    train_dataloader: SizedIterable[BatchState]
    val_dataloader: SizedIterable[BatchState]
    global_state: GlobalState
    hyper_parameters: HyperParameters
    save_dir: str
    current_step: int = 0
    scheduler: Optional[LRScheduler] = None

    @computed_field
    @property
    def total_steps(self) -> int:
        return len(self.train_dataloader) * (self.hyper_parameters.max_epoch - self.global_state.current_epoch + 1)

    @computed_field
    @property
    def remaining_epoch(self) -> int:
        return self.hyper_parameters.max_epoch - self.global_state.current_epoch

    @computed_field
    @property
    def grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        if self.hyper_parameters.amp: return torch.amp.GradScaler()
        return None

    @computed_field
    @property
    def on_fit(self) -> bool:
        if self.hyper_parameters.max_patient_num is None: return True
        if self.global_state.patience_count >= self.hyper_parameters.max_patient_num: return False
        return True

    def one_step(self) -> None:
        self.current_step += 1
        self.global_state.one_step()

    def one_epoch(self) -> None:
        self.current_step = 0
        self.global_state.one_epoch()
