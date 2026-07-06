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
├── distributed_utils.py             # DDP (分散学習) 用のヘルパー関数
└── engine_components/
    ├── console_reporter.py          # 学習開始時のサマリ表示
    ├── logging.py                   # チェックポイント・ログCSVの保存/読込
    └── tqdm_reporter.py             # 進捗バー表示（任意・実装必須ではありません）
```

## アーキテクチャの流れ

1. `main.py` で `HyperParameters` / `CoreComponents` / `FitContext` を組み立てる
2. `Engine(core_components).fit(fit_context)` を呼ぶと、
   - `fit()` の開始時に、今回のrunで使われた `HyperParameters` を `save_dir/hyper_parameters.json` へ
     静的ファイルとして記録する（`HyperParameters.save()` / `.load()`、DDP使用時はrank0のみ書き込む）
   - `_epoch_loop` が epoch ごとに train → val → checkpoint保存 を実行
   - 各バッチで `TaskModule.forward`（`mode`引数で `train_step` / `val_step` / `test_step` に分岐）を呼び出す
     （DDP使用時は `TaskModule` 全体をラップした `DistributedDataParallel` 経由で呼び出される）
   - `Metrics` が val/test の指標を蓄積・計算する
   - `Logging` がチェックポイントと `epoch_log.csv` を保存する
3. `Engine.test(...)` でテストデータに対する評価を行う
4. `Engine.system_check(...)` で本番前に少量バッチだけ流して動作確認を行う
   （`main.py` の例のように `fit()` の前に呼ぶ想定。学習前の `core_components` を必ず
   復元し、チェック用ディレクトリも必ず削除する ── `fit()` が途中で例外を送出しても同様）

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

    def to(self, device: str):
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

> **DDPとの関係**: `Engine` はDDP使用時、`TaskModule` 全体を1つの `DistributedDataParallel` で
> ラップします（サブモジュール単位ではラップしません）。DDPは「`forward()` の呼び出しを経由して
> 初めて勾配同期を正しく準備する」という制約があり、`train_step` / `val_step` / `test_step` は
> DDPからは認識されない独自メソッドのため、`Engine` は必ず `forward` 経由で呼び出します。
> `train_step` / `val_step` / `test_step` は出力の中身が異なる可能性が高いため、
> `self.training`（train/evalの2値）に頼るのではなく、`mode` 引数で明示的に分岐させています
> （`core_components.task_module` 自体は非ラップのまま保持されるため、`save` / `load` は
> 今まで通り変更不要です）。

```python
import os
from typing import Any, Literal, Optional

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

    def forward(
        self, batch_state: BatchState, fit_context: Optional[FitContext],
        mode: Literal["train", "val", "test"]
    ) -> BatchState:
        # Engineはこのモジュール自体をDistributedDataParallelでラップするため、
        # train_step/val_step/test_stepを直接呼ばず必ずこのforward経由で呼び出す。
        # train/val/testで出力が変わりうるため、mode引数で明示的に分岐する。
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

> **DDP使用時の注意**: 上記の `_correct` / `_total` はプロセスごとに独立して蓄積されるため、
> `compute()` の中で `distributed_utils.all_reduce_sum_` を使って全プロセス分を合計してから
> 指標を計算する必要があります（そうしないと各プロセスが自分の担当データ分だけの指標を返してしまいます）。
>
> ```python
> from distributed_utils import all_reduce_sum_
>
> def compute(self, mode: Literal["val", "test"]) -> dict[str, Any]:
>     correct = torch.tensor(self._correct[mode], dtype=torch.float64)
>     total = torch.tensor(self._total[mode], dtype=torch.float64)
>     all_reduce_sum_(correct)
>     all_reduce_sum_(total)
>     accuracy = (correct / total).item() if total > 0 else 0.0
>     return {"accuracy": accuracy}
> ```

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

以下の内容がそのまま `main.py` にテンプレートとして実装済みです。`train_df` / `val_df` の部分（実データへの
置き換え）と、`CustomizedDataset` / `dataset_collate_fn` / `TaskModule` / `Metrics` / `GlobalState._metric_update`
（上記1〜5の各TODO）を実装すれば、このまま学習を開始できます。
（`distributed_utils.setup_distributed()` / `cleanup_distributed()` の呼び出しは、
シングルプロセスで実行する場合は何もしないため、DDPを使わない場合もそのまま残しておいて構いません。
詳細は下記「DDP（分散学習）対応」を参照してください。）

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

    hyper_parameters = HyperParameters(
        max_epoch=10,
        batch_size=32,
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

    # get_dataloaderはDDP環境下では自動的にDistributedSamplerを使う
    train_dataloader = get_dataloader(train_dataset, hyper_parameters, shuffle=True, cpu_num_works=cpu_num_works)
    val_dataloader = get_dataloader(val_dataset, hyper_parameters, shuffle=False, cpu_num_works=cpu_num_works)

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
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
```

## DDP（分散学習）対応

**`main.py` は上の例のように `setup_distributed()` を呼んで `device` 文字列を組み立てるだけ**でよく、
既存の `HyperParameters` / `CoreComponents` の組み立て方は一切変えていません
（＝「できるだけ構造を維持したまま」DDP対応しています）。`TaskModule` 側は `forward` の追加が必要ですが、
`train_step` / `val_step` / `test_step` / `predict` / `save` / `load` 自体の中身は変更不要です。

> **`device` / `cpu_num_works` は `HyperParameters` ではなく `FitContext` のフィールドです**:
> `device`（例: `"cuda:1"`）はDDPのrankごとに異なりうる実行環境の情報、`cpu_num_works`（デフォルト `4`）も
> 実行環境（マシンのCPUコア数など）に依存する値であり、`max_epoch` / `batch_size` のような「実験の再現性を
> 記録すべきハイパーパラメータ」とは性質が異なります。`HyperParameters` は `save()` / `load()` で
> `hyper_parameters.json` として静的ログにする想定のため、これらを含めてしまうと、
> ログを読み込んで別プロセス/別マシンで再利用したときに古い（あるいは他rankの）値が
> 紛れ込む恐れがあります。そのため両方とも `HyperParameters` から外し、
> `FitContext.device` / `FitContext.cpu_num_works` として実行のたびに指定する設計にしています
> （`get_dataloader` にも `cpu_num_works` を独立した引数として渡します）。

### 何が変わったか

- **`distributed_utils.py`（新規）**: `setup_distributed()` / `cleanup_distributed()` / `barrier()` /
  `is_main_process()` / `all_reduce_sum_()` など、DDP まわりの共通処理をまとめたヘルパー。
  `torchrun` を使わずに単一プロセスで実行した場合はすべて no-op になるため、既存の使い方は壊れません。
- **`task_module.py`**: `TaskModule` に `forward(batch_state, fit_context, mode)` を追加しました。
  `Engine` は `TaskModule` 全体を1つの `DistributedDataParallel` でラップするため、
  `train_step` / `val_step` / `test_step` はDDPから認識されない独自メソッドとなり、`Engine` は
  必ず `forward` 経由で呼び出す必要があります。train/val/testで出力が変わりうることを明示するため、
  `self.training`（train/evalの2値）ではなく `mode`（`"train"` / `"val"` / `"test"`）引数で
  明示的に分岐させています。`fit_context` / `mode` にデフォルト値は持たせておらず、
  呼び出し側（`Engine`）が毎回明示的に指定する設計です（`test_step` のように `fit_context` が
  不要な場合は `None` を渡します）。`train_step` / `val_step` / `test_step` / `predict` /
  `save` / `load` 自体のインターフェースは変更していません。
- **`engine.py`**: `Engine` が `core_components.task_module` とは別に `self.model` を持つようにしました。
  `self.model` は `fit()` / `test()` の開始時に `_prepare_model()` で構築され、DDP環境なら
  `DistributedDataParallel(task_module, device_ids=[...])` で、そうでなければ元の `task_module`
  そのものです。`_train_step` は `self.model(batch_state, fit_context, mode="train")`、
  `_val_step` は `self.model(batch_state, fit_context, mode="val")`、
  `_test_step` は `self.model(batch_state, None, mode="test")` のように呼び出します。
  - `core_components.task_module`（生のモジュール）は保存・読込専用として常に非ラップのまま
    維持しているため、`CoreComponents.save/load` や `TaskModule.save/load` は今まで通り変更不要です
    （`state_dict()` に `module.` という接頭辞が付く、いわゆる DDP のチェックポイント問題を避けられます）。
  - チェックポイント保存・ログCSV書き込み・`TaskModule.save`・進捗バー表示・完了メッセージなど
    ファイル入出力やコンソール出力は `is_main_process()` でガードし、rank0のみが行うようにしました
    （複数プロセスが同じファイルへ同時に書き込むのを防ぐため）。
  - `_epoch_loop` の先頭で `model.train()` / 検証ループ前に `model.eval()` を呼ぶようにしました
    （BatchNorm/Dropoutの挙動を正しく切り替えるための一般的な作法で、DDPに限らず必要な変更です）。
  - `autocast(device_type=...)` に渡す値を `fit_context.device`（例: `"cuda:1"`）から
    `torch.device(...).type`（例: `"cuda"`）に変換するよう修正しました。DDPでは各プロセスが
    `"cuda:0"`, `"cuda:1"`, ... のようにインデックス付きのデバイス文字列を持つため、
    そのままでは `autocast` がエラーになる箇所を修正しています。
- **`data.py`**: `get_dataloader` は分散環境下では自動的に `DistributedSampler` を使い、
  各プロセスにデータを分割します（`shuffle` 引数は `DistributedSampler` 側に渡し、
  プロセス間でバッチ数を揃えるため `drop_last=True` にしています）。
  また `set_dataloader_epoch()` を追加し、`Engine._epoch_loop` がエポックごとに
  `DistributedSampler.set_epoch()` を呼び出してシャッフル系列を切り替えます。
- **`engine_components/*.py`**: `ConsoleReporter.show_fit_detail` / `Logging.save_checkpoint` /
  `TqdmReporter.init_*_bar` を `is_main_process()` でガードし、rank0のみが表示・書き込みを行うようにしました。

### あなたが実装する際に意識してほしい点

- **`task_module.py`**: `train_step` / `val_step` / `test_step` を実装するのに加えて、
  `forward` の `mode` 分岐（上記「3. `task_module.py`」の例）をそのまま利用してください。
  train/val/testで返す `batch_state` の中身（`loss` の有無、`preds` の意味など）が異なる設計を
  そのまま反映できます。
- **`metrics.py`**: 各プロセスは担当データ分の指標しか持っていないため、`compute()` の中で
  `distributed_utils.all_reduce_sum_` を使って全プロセス分を合計してから計算する必要があります
  （具体例は上記「4. `metrics.py`」の DDP 注意書きを参照）。
- **GANのように `discriminator` を1イテレーション中に複数回forwardする設計**（real/fake/G-loss用に
  複数回呼ぶなど）で、`discriminator` を更新しない `backward()`（G側の更新など）でも
  `discriminator` の勾配計算自体は発生するため、DDPは毎回all-reduceしようとします。
  無駄な通信や意図しない同期を避けたい場合は、`DistributedDataParallel.no_sync()`
  コンテキストの利用を検討してください（`train_step` 内で必要な箇所だけ囲む形になります）。

### 実行方法

```bash
# シングルGPU / CPU（今まで通り）
python main.py

# 単一ノード・複数GPU（例: 4GPU）
torchrun --standalone --nproc_per_node=4 main.py

# 複数ノード（例: 2ノード x 4GPU）
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=<マスターのIP>:29500 main.py
```

## 既知の注意点（実装前に確認しておくと良い箇所）

- ~~`engine_components/tqdm_reporter.py` の `set_posfitx` タイポ~~ → `set_postfix` に修正済み
- ~~`engine.py` の `system_check` 内の未定義変数 `save_dir` / `_epoch_loop` の引数不整合~~ → `system_check_save_dir` を使って `self.fit(...)` を呼ぶ形に修正済み
- ~~`configs.py`→`task_module.py`→`states.py`→`data.py`→`configs.py` の循環import~~ →
  型ヒントのみで実際には使われていない相互参照（`task_module.py`が`states`を、`data.py`が`configs`/`states`を
  参照する箇所）に `from __future__ import annotations` + `TYPE_CHECKING` を適用して遅延させ、
  実行時に必要な依存だけが残るよう解消済み（あわせて `SizedIterable` の定義を `data.py` から `states.py` へ移動）。
  この循環import自体が、これまで`main.py`はおろかどのファイルもimportできない状態を引き起こしていました。
- ~~`configs.CoreComponents` / `states.BatchState` / `states.FitContext` の `model_config` に
  `arbitrary_types_allowed=True` が無く、`TaskModule` / `torch.Tensor` / `SizedIterable` を
  フィールドに持てず pydantic のスキーマ生成でエラーになっていた~~ → 3クラスすべてに追加して解消済み。
- ~~`states.SizedIterable` が `Protocol[T]` のパラメータ化ジェネリックのまま `FitContext` のフィールド型に
  使われており、pydanticの `isinstance` 検証がパラメータ化ジェネリックを扱えずスキーマ生成エラーになっていた~~ →
  `@runtime_checkable` を付与し、フィールド定義では非パラメータ化の `SizedIterable` を使うよう修正済み。
- ~~`engine.py` の `system_check` に `try`/`finally` が無く、`self.fit(...)` が途中で例外を送出すると
  本番用の `self.core_components` が学習前の状態に復元されず、`.system_check/` ディレクトリも
  削除されないまま残っていた~~ → `try`/`finally` で復元・削除処理を必ず実行するよう修正済み
  （意図的に例外を起こして復元/削除されることを確認済み）。

これらは全ファイルの `import` とサンプルの `Engine.fit()` / `Engine.system_check()` 実行で動作確認済みです。
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
