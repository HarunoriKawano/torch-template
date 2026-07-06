# torch-template

PyTorch の学習・検証・テストループ（fit / test / system_check）を共通化したテンプレートです。
タスク固有の部分（モデル、データセット、損失計算、評価指標）だけを実装すれば、
学習ループやチェックポイント保存・ログ出力・進捗バー表示などは `Engine` が面倒を見ます。

以下の実装例は、シンプルな **GAN**（`preprocessing` / `generator` / `discriminator` / `criterion` を
持つ `TaskModule`）を想定した例です。実際のタスクに合わせて置き換えてください。

## ディレクトリ構成

```
.
├── main.py                          # エントリーポイント
├── templates/
│   └── hyper_parameters.json        # HyperParametersの静的設定ファイル（main.pyが読み込む）
├── configs.py                       # HyperParameters, CoreComponents（モデル・optimizerの保存/読込）
├── states.py                        # BatchState, GlobalState, FitContext
├── data.py                          # Dataset, collate_fn, DataLoader 生成
├── task_module.py                   # TaskModule（モデルのラッパー：学習/検証/推論ロジック）
├── metrics.py                       # Metrics（評価指標の蓄積・計算）
├── engine.py                        # Engine（fit / test / system_check ループ本体）
├── distributed_utils.py             # DDP (分散学習) 用のヘルパー関数
└── engine_components/
    ├── console_reporter.py          # 学習開始時のサマリ表示
    ├── logging.py                   # チェックポイント・ログCSVの保存/読込
    └── tqdm_reporter.py             # 進捗バー表示（任意）
```

## アーキテクチャの流れ

1. `main.py` で `HyperParameters` / `CoreComponents` / `FitContext` を組み立てる
2. `Engine(core_components).system_check(fit_context)` で少量バッチだけ流して動作確認を行う
3. `Engine.fit(fit_context)` で学習を実行する
   - `_epoch_loop` が epoch ごとに train → val → checkpoint保存 を実行
   - 各バッチで `TaskModule.forward`（`mode`引数で `train_step` / `val_step` / `test_step` に分岐）を呼び出す
   - `Metrics` が val/test の指標を蓄積・計算する
   - `Logging` がチェックポイントと `epoch_log.csv` を保存する
4. `Engine.test(...)` でテストデータに対する評価を行う

## 使用方法（実装する順番）

このテンプレートは以下の順番で実装していくと、手戻りが少なくスムーズに進められます。

1. オリジナルのモデルを定義する
2. `task_module.py` — モデルを `TaskModule` でラップしてフローをまとめる
3. `data.py` — データセットを作成する
4. `train` / `val` / `test` の入出力を確認した上で `states.py` — `BatchState` を定義する
5. `metrics.py` — 評価指標を実装する
6. `states.py` — `GlobalState._metric_update` を実装する
7. `templates/hyper_parameters.json` を編集し、`main.py` を実データに差し替えて実行する

### 1. オリジナルのモデルを定義する

まずは torch-template 固有の型（`BatchState` など）を一切意識せず、普段どおり PyTorch でモデルを実装します。
GAN であれば `Generator` / `Discriminator` のように、素の `nn.Module` として書くだけです。

```python
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 28 * 28), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### 2. `task_module.py` — モデルを `TaskModule` でラップする

1. で定義したモデルを `TaskModule` にまとめ、損失計算〜推論までのフローを実装します。
`save` / `load` は、学習済みパラメータとして保存・復元が必要な **`generator` と `discriminator` のみ**
を対象にします（`preprocessing` はパラメータを持たない前処理、`criterion` は損失関数でパラメータを
持たないため対象外）。

この時点では `batch_state.inputs` / `batch_state.preds` のように、まだ正式には定義していない
`BatchState` のフィールドを前提にコードを書くことになります。「`train_step` は何を受け取り何を返すか」
「`val_step` は（`preds` など）追加で何を返す必要があるか」をここで洗い出しておくと、4. で
`BatchState` を定義するときに迷いません。

`train_step` / `val_step` / `test_step` は出力の中身が異なるため、`forward` は `mode` 引数
（`"train"` / `"val"` / `"test"`）で明示的に分岐させています。

```python
import os
from typing import Any, Literal, Optional

import torch
from torch import nn

from states import BatchState, FitContext
# 1. で定義したモデル
# from models import Generator, Discriminator


class TaskModule(nn.Module):
    generator_file_name = "generator.pth"
    discriminator_file_name = "discriminator.pth"

    def __init__(self):
        super().__init__()
        self.preprocessing = Preprocessing()   # 例: 正規化など、パラメータを持たない前処理
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, batch_state: BatchState, fit_context: Optional[FitContext],
        mode: Literal["train", "val", "test"]
    ) -> BatchState:
        if mode == "train":
            return self.train_step(batch_state, fit_context)
        if mode == "val":
            return self.val_step(batch_state)
        if mode == "test":
            return self.test_step(batch_state)
        raise ValueError(f"unknown mode: {mode}")

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

### 3. `data.py` — データセットを作成する

`__init__` では pandas の `DataFrame` を受け取り、`__getitem__` では `iloc` で1行ずつ値を取り出します。
この段階では、まだ `BatchState` への変換（`dataset_collate_fn`）は実装しません。

```python
import pandas as pd
import torch
from torch.utils.data import Dataset


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
```

### 4. `train` / `val` / `test` の入出力を確認して `states.py` — `BatchState` を定義する

2. で書いた `train_step` / `val_step` / `test_step` が実際に何を受け取り、何を返すか
（例: `train_step` は `inputs` / `labels` を受け取り `loss` を更新して返す、`val_step` はさらに
`preds` も返す）を確認し、3. で作った `Dataset` が返す値と合わせて `BatchState` のフィールドを確定させます。
`Dataset` の出力を `BatchState` に変換する `dataset_collate_fn`（`data.py`）もここで実装します。

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

    def to(self, device: str):
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        if self.preds is not None:
            self.preds = self.preds.to(device)
```

```python
# data.py に追加する
import torch

from states import BatchState


def dataset_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> BatchState:
    inputs, labels = zip(*batch)
    return BatchState(
        inputs=torch.stack(inputs),
        labels=torch.stack(labels),
    )
```

`get_dataloader` / `ShortDataLoader` はそのまま利用できます。

### 5. `metrics.py` — 評価指標を実装する

```python
from typing import Literal, Any

import torch

from states import BatchState
from distributed_utils import all_reduce_sum_


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
        # DDP使用時、_correct/_totalはプロセスごとに別々の値を持つため、
        # 全プロセス分を合計してから指標を計算する
        correct = torch.tensor(self._correct[mode], dtype=torch.float64)
        total = torch.tensor(self._total[mode], dtype=torch.float64)
        all_reduce_sum_(correct)
        all_reduce_sum_(total)
        accuracy = (correct / total).item() if total > 0 else 0.0
        return {"accuracy": accuracy}

    def reset(self) -> None:
        self._correct = {"val": 0, "test": 0}
        self._total = {"val": 0, "test": 0}
```

### 6. `states.py` — `GlobalState._metric_update`

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

### 7. `templates/hyper_parameters.json` と `main.py`

`HyperParameters`（実験の再現性を記録すべき値）は `templates/hyper_parameters.json` に静的ファイルとして
定義しておき、`main.py` から `HyperParameters.load(...)` で読み込みます。値を変えたいときはこのJSONを
編集するだけで済み、コードを触る必要はありません。

```json
{
  "max_epoch": 50,
  "batch_size": 64,
  "amp": "bfloat16",
  "max_patient_num": 10
}
```

`main.py` は以下の内容がテンプレートとして実装済みです。`train_df` / `val_df` / `test_df` の部分を
実データに置き換え、1〜6を実装すればそのまま学習・評価を開始できます。

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
from distributed_utils import setup_distributed, cleanup_distributed


def main():
    rank, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    cpu_num_works = 4

    hyper_parameters = HyperParameters.load("templates/hyper_parameters.json")

    # TODO: 実データ（CSV読み込みなど）に置き換える
    train_df = pd.DataFrame({
        "x": list(np.random.randn(1000, 1, 28, 28)),
        "label": np.random.randint(0, 10, size=1000),
    })
    val_df = pd.DataFrame({
        "x": list(np.random.randn(200, 1, 28, 28)),
        "label": np.random.randint(0, 10, size=200),
    })
    test_df = pd.DataFrame({
        "x": list(np.random.randn(200, 1, 28, 28)),
        "label": np.random.randint(0, 10, size=200),
    })
    train_dataset = CustomizedDataset(train_df)
    val_dataset = CustomizedDataset(val_df)
    test_dataset = CustomizedDataset(test_df)

    # get_dataloaderはDDP環境下では自動的にDistributedSamplerを使う
    train_dataloader = get_dataloader(train_dataset, hyper_parameters, shuffle=True, cpu_num_works=cpu_num_works)
    val_dataloader = get_dataloader(val_dataset, hyper_parameters, shuffle=False, cpu_num_works=cpu_num_works)
    test_dataloader = get_dataloader(test_dataset, hyper_parameters, shuffle=False, cpu_num_works=cpu_num_works)

    task_module = TaskModule()
    optimizer = Adam(task_module.parameters(), lr=1e-3)
    core_components = CoreComponents(task_module=task_module, optimizer=optimizer)

    global_state = GlobalState(best_metric=0.0)
    fit_context = FitContext(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        global_state=global_state,
        hyper_parameters=hyper_parameters,
        device=device,
        cpu_num_works=cpu_num_works,
        save_dir="./checkpoints",
    )

    try:
        engine = Engine(core_components)
        engine.system_check(fit_context)
        engine.fit(fit_context)

        metrics = engine.test(test_dataloader, hyper_parameters, device)
        print(metrics.compute("test"))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

## DDP（分散学習）対応

`main.py` で `setup_distributed()` を呼び、`device` 文字列（例: `"cuda:0"`）を組み立てるだけで、
`Engine` が以下を自動的に行います。

- `TaskModule` を `DistributedDataParallel` でラップする
- `get_dataloader` が `DistributedSampler` でデータをプロセスごとに分割する
- チェックポイント保存・ログ出力・進捗バー表示はrank0のみが行う

`device` / `cpu_num_works` はDDPのrankや実行環境ごとに異なりうる値のため、`HyperParameters` ではなく
`FitContext` のフィールドとして実行のたびに指定します。

### 実行方法

```bash
# シングルGPU / CPU
python main.py

# 単一ノード・複数GPU（例: 4GPU）
torchrun --standalone --nproc_per_node=4 main.py

# 複数ノード（例: 2ノード x 4GPU）
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=<マスターのIP>:29500 main.py
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行

```bash
python main.py
```

（1〜6のTODOを実装してから実行してください）
