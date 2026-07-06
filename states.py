from __future__ import annotations

from typing import Optional, Protocol, Iterator, TypeVar, TYPE_CHECKING, runtime_checkable

from pydantic import BaseModel, ConfigDict
import torch
from pydantic import computed_field
from torch.optim.lr_scheduler import LRScheduler

from configs import HyperParameters

if TYPE_CHECKING:
    # Metricsはメソッド引数の型ヒントとしてのみ使われ、フィールドの型ではないため、
    # 実行時のimportは不要（states<->metricsの循環importを避けるため遅延させている）
    from metrics import Metrics

T = TypeVar('T')

@runtime_checkable
class SizedIterable(Protocol[T]):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T]:
        ...

# TODO バッチ内で完結する値を定義する(入力値やラベルなど)
class BatchState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # pydanticのisinstance検証はパラメータ化ジェネリックを扱えないため、
    # 型ヒント上はSizedIterable[BatchState]としたいところだが、フィールドでは非パラメータ化で宣言する
    train_dataloader: SizedIterable
    val_dataloader: SizedIterable
    global_state: GlobalState
    hyper_parameters: HyperParameters
    # device/cpu_num_worksは実行環境ごとに異なりうる情報であり、
    # 実験の再現性を記録するHyperParametersには含めない
    device: str
    save_dir: str
    cpu_num_works: int = 4
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
