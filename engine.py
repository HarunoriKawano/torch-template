import os
import shutil
import copy

import torch
from torch.amp import autocast

from states import FitContext, BatchState
from configs import CoreComponents, HyperParameters
from metrics import Metrics
from data import ShortDataLoader, SizedIterable
from engine_components.console_reporter import ConsoleReporter
from engine_components.logging import Logging
from engine_components.tqdm_reporter import TqdmReporter


class Engine:
    system_check_iteration_num: int = 5
    system_check_save_dir: str = "./.system_check/"

    def __init__(self, core_components: CoreComponents) -> None:
        self.core_components = core_components
        self.metrics = Metrics()
        self.console_reporter = ConsoleReporter()
        self.logging = Logging()
        self.tqdm_reporter = TqdmReporter()

    def fit(self, fit_context: FitContext) -> None:
        # fit init
        os.makedirs(fit_context.save_dir, exist_ok=True)
        self.logging.load_checkpoint(self.core_components, fit_context)
        self.console_reporter.show_fit_detail(self.core_components, fit_context)

        for _ in range(fit_context.remaining_epoch):
            if not fit_context.on_fit:
                break
            self._epoch_loop(fit_context)

    def _epoch_loop(self, fit_context: FitContext) -> None:
        # train loop
        self.tqdm_reporter.init_train_bar(fit_context)
        for batch_state in fit_context.train_dataloader:
            self._train_step(fit_context, batch_state)
        self.tqdm_reporter.reset_bar()
        fit_context.one_epoch()

        # val loop
        self.metrics.reset()
        self.tqdm_reporter.init_val_bar(fit_context)
        for batch_state in fit_context.val_dataloader:
            self._val_step(fit_context, batch_state)
        self.tqdm_reporter.reset_bar()

        # checkpoint save
        self.logging.save_checkpoint(self.core_components, fit_context, self.metrics)

        # check result
        if fit_context.global_state.check_metric(self.metrics):
            self.core_components.task_module.save(fit_context.save_dir)

    def _train_step(self, fit_context: FitContext, batch_state: BatchState) -> None:
        # init
        self.tqdm_reporter.on_train_batch(fit_context)
        self.core_components.optimizer.zero_grad(set_to_none=True)
        batch_state.to(fit_context.hyper_parameters.device)

        # step
        dtype = getattr(torch, fit_context.hyper_parameters.amp) if fit_context.hyper_parameters.amp is not None else None
        with autocast(device_type=fit_context.hyper_parameters.device, enabled=bool(fit_context.hyper_parameters.amp), dtype=dtype):
            batch_state = self.core_components.task_module.train_step(batch_state, fit_context)

        if batch_state.loss is None:
            raise TypeError("lossが更新されていません。")
        if fit_context.grad_scaler:
            fit_context.grad_scaler.scale(batch_state.loss).backward()
            fit_context.grad_scaler.step(self.core_components.optimizer)
            fit_context.grad_scaler.update()
        else:
            batch_state.loss.backward()
            self.core_components.optimizer.step()

        if fit_context.scheduler:
            fit_context.scheduler.step()

        # post process
        batch_state.to("cpu")
        fit_context.one_step()
        self.tqdm_reporter.set_train_metrics(batch_state)

    def _val_step(self, fit_context: FitContext, batch_state: BatchState) -> None:
        # init
        self.tqdm_reporter.on_val_batch(fit_context)
        batch_state.to(fit_context.hyper_parameters.device)

        # step
        dtype = getattr(torch, fit_context.hyper_parameters.amp) if fit_context.hyper_parameters.amp is not None else None
        with autocast(device_type=fit_context.hyper_parameters.device, enabled=bool(fit_context.hyper_parameters.amp), dtype=dtype):
            batch_state = self.core_components.task_module.val_step(batch_state)

        # post process
        batch_state.to("cpu")
        self.metrics.update(batch_state, "val")
        self.tqdm_reporter.set_val_metrics(batch_state)

    def test(self, test_dataloader: SizedIterable[BatchState], hyper_parameters: HyperParameters) -> Metrics:
        self.metrics.reset()

        # test loop
        self.tqdm_reporter.init_test_bar(test_dataloader)
        for batch_state in test_dataloader:
            self._test_step(hyper_parameters, batch_state)
        self.tqdm_reporter.reset_bar()

        return self.metrics

    def _test_step(self, hyper_parameters: HyperParameters, batch_state: BatchState) -> None:
        # init
        self.tqdm_reporter.on_test_batch()
        batch_state.to(hyper_parameters.device)

        # step
        dtype = getattr(torch, hyper_parameters.amp) if hyper_parameters.amp is not None else None
        with autocast(device_type=hyper_parameters.device, enabled=bool(hyper_parameters.amp), dtype=dtype):
            batch_state = self.core_components.task_module.val_step(batch_state)

        # post process
        batch_state.to("cpu")
        self.metrics.update(batch_state, "test")
        self.tqdm_reporter.set_test_metrics(batch_state)

    def system_check(self, fit_context: FitContext) -> None:
        copy_fit_context = copy.deepcopy(fit_context)
        copy_core_components = copy.deepcopy(self.core_components)

        copy_fit_context.hyper_parameters = HyperParameters(
            max_epoch=1, batch_size=copy_fit_context.hyper_parameters.batch_size,
            device=copy_fit_context.hyper_parameters.device,
            cpu_num_works=copy_fit_context.hyper_parameters.cpu_num_works
        )
        copy_fit_context.train_dataloader = ShortDataLoader(copy_fit_context.train_dataloader, self.system_check_iteration_num)
        copy_fit_context.val_dataloader = ShortDataLoader(copy_fit_context.val_dataloader, self.system_check_iteration_num)
        copy_fit_context.save_dir = self.system_check_save_dir

        self.fit(copy_fit_context)

        shutil.rmtree(self.system_check_save_dir)
        self.core_components = copy_core_components

        print("システムチェック成功.")



