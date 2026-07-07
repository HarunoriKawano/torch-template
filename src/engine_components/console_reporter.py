from src.configs import CoreComponents
from src.states import FitContext
from src.distributed_utils import is_main_process


class ConsoleReporter:
    @staticmethod
    def _strong_print(strings: list[str]):
        if not strings:
            return
        max_length = max([len(string) for string in strings])
        print(f"\n{'=' * (max_length + 4)}")
        for string in strings:
            print(f" {string:<{max_length}} ")
        print(f"{'=' * (max_length + 4)}\n")

    def show_fit_detail(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        if not is_main_process():
            return

        num_params: int = 0
        for p in core_components.task_module.parameters():
            num_params += p.numel()

        self._strong_print([
            "Training Start",
            f"Device:      {fit_context.device}",
            f"Remaining epochs: {fit_context.remaining_epoch}",
            f"Steps per epoch:  {len(fit_context.train_dataloader)}",
            f"Total steps:      {fit_context.remaining_epoch * len(fit_context.train_dataloader)}",
            f"Model parameters: {num_params}",
            f"Batch size:       {fit_context.hyper_parameters.batch_size}",
        ])
