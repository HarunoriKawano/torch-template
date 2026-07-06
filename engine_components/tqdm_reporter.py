from tqdm import tqdm

from data import SizedIterable
from states import FitContext, BatchState
from distributed_utils import is_main_process

class TqdmReporter:
    def __init__(self):
        self.pbar = None

    def init_train_bar(self, fit_context: FitContext) -> None:
        """エポック開始時に新しいプログレスバーを生成する（chief processのみ表示）"""
        if not is_main_process():
            return
        # leave=True にすると、終わったバーが画面に残ります（学習履歴として見やすい）
        self.pbar = tqdm(
            total=len(fit_context.train_dataloader),
            desc=f"Epoch {fit_context.global_state.current_epoch}",
            leave=True
        )

    def on_train_batch(self, fit_context: FitContext) -> None:
        """バッチ終了時にバーを1つ進め、Lossの数値を右側に表示する"""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f'[Train] [Epoch {fit_context.train_dataloader}/{fit_context.hyper_parameters.max_epoch}]')

    # TODO progress barの右側に表示する値を決める
    def set_train_metrics(self, batch_state: BatchState) -> None:
        if self.pbar is not None:
            self.pbar.set_postfix({"loss": f"{batch_state.loss}"})

    def init_val_bar(self, fit_context: FitContext) -> None:
        """エポック開始時に新しいプログレスバーを生成する（chief processのみ表示）"""
        if not is_main_process():
            return
        # leave=True にすると、終わったバーが画面に残ります（学習履歴として見やすい）
        self.pbar = tqdm(
            total=len(fit_context.val_dataloader),
            desc=f"Epoch {fit_context.global_state.current_epoch}",
            leave=True
        )

    def on_val_batch(self, fit_context: FitContext) -> None:
        """バッチ終了時にバーを1つ進め、Lossの数値を右側に表示する"""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f'[Val] [Epoch {fit_context.train_dataloader}/{fit_context.hyper_parameters.max_epoch}]')

    # TODO progress barの右側に表示する値を決める
    def set_val_metrics(self, batch_state: BatchState) -> None:
        if self.pbar is not None:
            self.pbar.set_postfix({"loss": f"{batch_state.loss}"})

    def init_test_bar(self, test_dataloader: SizedIterable[BatchState]) -> None:
        """エポック開始時に新しいプログレスバーを生成する（chief processのみ表示）"""
        if not is_main_process():
            return
        # leave=True にすると、終わったバーが画面に残ります（学習履歴として見やすい）
        self.pbar = tqdm(
            total=len(test_dataloader),
            leave=True
        )

    def on_test_batch(self) -> None:
        """バッチ終了時にバーを1つ進め、Lossの数値を右側に表示する"""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f'[Test]')

    # TODO progress barの右側に表示する値を決める
    def set_test_metrics(self, batch_state: BatchState) -> None:
        if self.pbar is not None:
            self.pbar.set_postfix({"loss": f"{batch_state.loss}"})

    def reset_bar(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
