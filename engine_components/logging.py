import pandas as pd
import os
from pathlib import Path

from states import FitContext
from metrics import Metrics
from configs import CoreComponents

class Logging:
    epoch_log_file_name = "epoch_log.csv"
    core_components_file_name = "core_components.pth"
    global_state_file_name = "global_state.pth"

    def load_checkpoint(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        components_checkpoint_path = Path(os.path.join(fit_context.save_dir, self.core_components_file_name))
        state_checkpoint_path = Path(os.path.join(fit_context.save_dir, self.global_state_file_name))
        if components_checkpoint_path.exists() and state_checkpoint_path.exists():
            core_components.load(str(components_checkpoint_path), fit_context.hyper_parameters.device)
            fit_context.global_state.load(str(state_checkpoint_path))
            print(f"Checkpoint found. Resuming from epoch {fit_context.remaining_epoch}/{fit_context.hyper_parameters.max_epoch}.\n")
        else:
            print("No checkpoint found. Starting training from scratch.\n")

    def save_checkpoint(self, core_components: CoreComponents, fit_context: FitContext, metrics: Metrics) -> None:
        computed_metrics = metrics.compute("val")
        epoch_log_df = pd.DataFrame([computed_metrics])
        epoch_log_df.to_csv(
            os.path.join(fit_context.save_dir, self.epoch_log_file_name),
            encoding="utf-8",
            mode="w" if fit_context.global_state.current_epoch == 1 else "a",
            index=False,
            header=fit_context.global_state.current_epoch == 1
        )
        core_components.save(os.path.join(fit_context.save_dir, self.core_components_file_name))
        fit_context.global_state.save(os.path.join(fit_context.save_dir, self.global_state_file_name))
