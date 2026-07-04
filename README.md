# torch-template

PyTorch の学習・検証・テストループ（fit / test / system_check）を共通化したテンプレートです。
タスク固有の部分（データセット、モデル、損失計算、評価指標）だけを実装すれば、
学習ループやチェックポイント保存・ログ出力・進捗バー表示などは `Engine` が面倒を見ます。

以下の実装例は、シンプルな**画像分類タスク**（入力画像 `x` とラベル `y` から `CrossEntropyLoss` で学習し、
正解率を評価指標とする）を想定した例です。実際のタスクに合わせて置き換えてください。
ただし `TaskModule`（3.）だけは、複数のサブモジュールを持つ場合の `save` / `load` の書き方を示すため、
あえて **GAN** を想定した例にしています。

## ディレクトリ構成

```
.
├── main.py                          # エントリーポイント（未実装：ここから学習/推論を呼び出す）
├── configs.py                       # HyperParameters, CoreComponents（モデル・optimizerの保存/読込）
├── states.py                        # BatchState, GlobalState, FitContext
├── data.py                          # Dataset, collate_fn, DataLoader 生成
├── task_module.py                   # TaskModule（モデルのラッパー：学習/検証/推論ロジック）
├── metrics.py                       # Metrics（評価指標の蓄積・計算）
├── engine.py                        # Engine（fit / test / system_check ループ本体）
└── engine_components/
    ├── console_reporter.py          # 学習開始時のサマリ表示
    ├── logging.py                   # チェックポイント・ログCSVの保存/読込
    └── tqdm_reporter.py             # 進捗バー表示（任意・実装必須ではありません）
```

## アーキテクチャの流れ

1. `main.py` で `HyperParameters` / `CoreComponents` / `FitContext` を組み立てる
2. `Engine(core_components).fit(fit_context)` を呼ぶと、
   - `_epoch_loop` が epoch ごとに train → val → checkpoint保存 を実行
   - 各バッチで `TaskModule.train_step` / `val_step` を呼び出す
   - `Metrics` が val/test の指標を蓄積・計算する
   - `Logging` がチェックポイントと `epoch_log.csv` を保存する
3. `Engine.test(...)` でテストデータに対する評価を行う
4. `Engine.system_check(...)` で本番前に少量バッチだけ流して動作確認を行う

## 実装が必要な項目（推奨する実装順・実装例つき）

### 1. `states.py` — `BatchState`

バッチ単位で完結するデータ（入力・ラベルなど）を定義します。

- フィールド定義: 入力テンソルやラベルなど、タスクに応じたフィールドを追加
- `to(self, device)`: 保持しているテンソルを指定デバイスへ移動する処理を実装

```python
from typing import Optional
from pydantic import BaseModel, ConfigDict
import torch


class BatchState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: Optional[torch.Tensor] = None
    inputs: torch.Tensor
    labels: torch.Tensor
    # 予測値は val/test 時に Metrics へ渡すために保持しておく
    preds: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        if self.preds is not None:
            self.preds = self.preds.to(device)
```

### 2. `data.py` — `CustomizedDataset` / `dataset_collate_fn`

`__init__` では pandas の `DataFrame` を受け取り、`__getitem__` では `iloc` で1行ずつ値を取り出します。

```python
import pandas as pd
import torch
from torch.utils.data import Dataset

from states import BatchState


class CustomizedDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # 例: 特徴量列 "x" とラベル列 "label" を持つ DataFrame
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x = torch.tensor(row["x"], dtype=torch.float32)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x, y


def dataset_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> BatchState:
    inputs, labels = zip(*batch)
    return BatchState(
        inputs=torch.stack(inputs),
        labels=torch.stack(labels),
    )
```

`get_dataloader` / `ShortDataLoader` はそのまま利用できます。

### 3. `task_module.py` — `TaskModule`

`TaskModule` は「入力から出力までを完結させる end-to-end のモデル」をまるごとラップするクラスです。
例えば GAN であれば `preprocessing` / `generator` / `discriminator` / `criterion` のように
複数のサブモジュールを持たせます。このとき `save` / `load` は、学習済みパラメータとして
保存・復元が必要な **`generator` と `discriminator` のみ** を対象にします
（`preprocessing` はパラメータを持たない前処理、`criterion` は損失関数でパラメータを持たないため対象外）。

```python
import os
from typing import Any

import torch
from torch import nn

from states import BatchState, FitContext


class TaskModule(nn.Module):
    generator_file_name = "generator.pth"
    discriminator_file_name = "discriminator.pth"

    def __init__(self):
        super().__init__()
        self.preprocessing = Preprocessing()   # 例: 正規化など、パラメータを持たない前処理
        self.generator = Generator()           # 例: 入力からデータを生成するモデル
        self.discriminator = Discriminator()   # 例: 本物/生成物を判定するモデル
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, batch_state: BatchState, fit_context: FitContext) -> BatchState:
        x = self.preprocessing(batch_state.inputs)
        fake = self.generator(x)

        real_logits = self.discriminator(batch_state.inputs)
        fake_logits = self.discriminator(fake.detach())
        d_loss = (
            self.criterion(real_logits, torch.ones_like(real_logits))
            + self.criterion(fake_logits, torch.zeros_like(fake_logits))
        )

        g_logits = self.discriminator(fake)
        g_loss = self.criterion(g_logits, torch.ones_like(g_logits))

        # Engine は batch_state.loss を1つだけ backward するため、
        # generator/discriminator を別々に最適化したい場合は、ここで個別に
        # optimizer.step() まで完結させる、もしくは合算した loss を返すなど
        # 設計に応じて調整する
        batch_state.loss = g_loss + d_loss
        return batch_state

    def val_step(self, batch_state: BatchState) -> BatchState:
        with torch.no_grad():
            x = self.preprocessing(batch_state.inputs)
            batch_state.preds = self.generator(x)
        return batch_state

    # test_step は val_step を再利用するのでそのままでよい

    def predict(self, x: torch.Tensor) -> Any:
        with torch.no_grad():
            return self.generator(self.preprocessing(x))

    def save(self, save_dir: str) -> None:
        # 保存対象は学習済みパラメータを持つ generator / discriminator のみ
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(save_dir, self.generator_file_name))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, self.discriminator_file_name))

    def load(self, save_dir: str) -> None:
        self.generator.load_state_dict(
            torch.load(os.path.join(save_dir, self.generator_file_name), map_location="cpu")
        )
        self.discriminator.load_state_dict(
            torch.load(os.path.join(save_dir, self.discriminator_file_name), map_location="cpu")
        )
```

> 上記は `save` / `load` の考え方を示すための GAN 例です。`batch_state.preds` の中身
> （分類タスクなら予測ラベル、GANなら生成データ）や `Metrics` の計算方法は、
> 実際に組み合わせるタスクに合わせて設計してください。

### 4. `metrics.py` — `Metrics`

```python
from typing import Literal, Any

import torch

from states import BatchState


class Metrics:
    """
    評価指標を計算するためのインターフェース。
    バッチごとに状態を蓄積し、エポックの最後に計算・リセットする。
    """

    def __init__(self):
        self._correct: dict[str, int] = {"val": 0, "test": 0}
        self._total: dict[str, int] = {"val": 0, "test": 0}

    def update(self, batch_state: BatchState, mode: Literal["val", "test"]) -> None:
        self._correct[mode] += (batch_state.preds == batch_state.labels).sum().item()
        self._total[mode] += batch_state.labels.numel()

    def compute(self, mode: Literal["val", "test"]) -> dict[str, Any]:
        total = self._total[mode]
        accuracy = self._correct[mode] / total if total > 0 else 0.0
        return {"accuracy": accuracy}

    def reset(self) -> None:
        self._correct = {"val": 0, "test": 0}
        self._total = {"val": 0, "test": 0}
```

### 5. `states.py` — `GlobalState._metric_update`

`Metrics` の計算結果を受け取り、ベストスコアを更新したかどうかを判定して `bool` を返す処理を実装します。
この結果に応じて `patience_count` の増減や `TaskModule.save` の呼び出しが行われます。

```python
def _metric_update(self, metric: "Metrics") -> bool:
    computed = metric.compute("val")
    accuracy = computed["accuracy"]
    if accuracy > self.best_metric:
        self.best_metric = accuracy
        return True
    return False
```

### 6. `main.py`

現在 PyCharm のサンプルコードのままなので、以下を組み立てるエントリーポイントに書き換えます。

```python
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from configs import HyperParameters, CoreComponents
from data import CustomizedDataset, get_dataloader
from states import GlobalState, FitContext
from task_module import TaskModule
from engine import Engine

def main():
    hyper_parameters = HyperParameters(
        max_epoch=10,
        batch_size=32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cpu_num_works=2,
    )

    # TODO: 実データ（CSV読み込みなど）に置き換える
    train_df = pd.DataFrame({
        "x": list(np.random.randn(1000, 1, 28, 28)),
        "label": np.random.randint(0, 10, size=1000),
    })
    val_df = pd.DataFrame({
        "x": list(np.random.randn(200, 1, 28, 28)),
        "label": np.random.randint(0, 10, size=200),
    })
    train_dataset = CustomizedDataset(train_df)
    val_dataset = CustomizedDataset(val_df)

    train_dataloader = get_dataloader(train_dataset, hyper_parameters, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, hyper_parameters, shuffle=False)

    task_module = TaskModule().to(hyper_parameters.device)
    optimizer = Adam(task_module.parameters(), lr=1e-3)
    core_components = CoreComponents(task_module=task_module, optimizer=optimizer)

    global_state = GlobalState(best_metric=0.0)
    fit_context = FitContext(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        global_state=global_state,
        hyper_parameters=hyper_parameters,
        save_dir="./checkpoints",
    )

    engine = Engine(core_components)
    engine.fit(fit_context)


if __name__ == "__main__":
    main()
```

## 既知の注意点（実装前に確認しておくと良い箇所）

- ~~`engine_components/tqdm_reporter.py` の `set_posfitx` タイポ~~ → `set_postfix` に修正済み
- ~~`engine.py` の `system_check` 内の未定義変数 `save_dir` / `_epoch_loop` の引数不整合~~ → `system_check_save_dir` を使って `self.fit(...)` を呼ぶ形に修正済み

上記以外にも、既存コードの中に動作を妨げる可能性のある箇所が残っています。実装を進める際に合わせて確認してください。

- `engine_components/tqdm_reporter.py`（任意で使う場合のみ）: `on_train_batch` / `on_val_batch` の `set_description` で `fit_context.train_dataloader` を文字列展開している箇所は、意図としては現在のエポック数（`fit_context.global_state.current_epoch`）を表示したいものと思われます。

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行

```bash
python main.py
```

（`main.py` の実装が完了してから実行してください）
