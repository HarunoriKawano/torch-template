import pandas as pd
import os
from pathlib import Path

from states import FitContext
from metrics import Metrics
from configs import CoreComponents
from distributed_utils import is_main_process

class Logging:
    epoch_log_file_name = "epoch_log.csv"
    core_components_file_name = "core_components.pth"
    global_state_file_name = "global_state.pth"
    hyper_parameters_file_name = "hyper_parameters.json"
    model_parameters_file_name = "model_parameters.json"

    def save_parameters(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        # 今回のrunで使われたHyperParametersを静的ファイルとして記録する（rank0のみ）
        if not is_main_process():
            return

        fit_context.hyper_parameters.save(os.path.join(fit_context.save_dir, self.hyper_parameters_file_name))
        core_components.task_module.model_parameters.save(os.path.join(fit_context.save_dir, self.model_parameters_file_name))


    def load_checkpoint(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        components_checkpoint_path = Path(os.path.join(fit_context.save_dir, self.core_components_file_name))
        state_checkpoint_path = Path(os.path.join(fit_context.save_dir, self.global_state_file_name))
        if components_checkpoint_path.exists() and state_checkpoint_path.exists():
            core_components.load(str(components_checkpoint_path), fit_context.device)
            fit_context.global_state.load(str(state_checkpoint_path))
            print(f"Checkpoint found. Resuming from epoch {fit_context.remaining_epoch}/{fit_context.hyper_parameters.max_epoch}.\n")
        else:
            print("No checkpoint found. Starting training from scratch.\n")

    def save_checkpoint(self, core_components: CoreComponents, fit_context: FitContext, metrics: Metrics) -> None:
        # チェックポイント/ログの書き込みはchief process (rank0) のみが行う
        if not is_main_process():
            return
        train_computed_metrics = metrics.compute("train")
        computed_metrics = metrics.compute("val")
        epoch_log_df = pd.DataFrame([train_computed_metrics | computed_metrics])
        epoch_log_df.to_csv(
            os.path.join(fit_context.save_dir, self.epoch_log_file_name),
            encoding="utf-8",
            mode="w" if fit_context.global_state.current_epoch == 1 else "a",
            index=False,
            header=fit_context.global_state.current_epoch == 1
        )
        core_components.save(os.path.join(fit_context.save_dir, self.core_components_file_name))
        fit_context.global_state.save(os.path.join(fit_context.save_dir, self.global_state_file_name))
