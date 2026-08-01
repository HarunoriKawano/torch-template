import os
import shutil
import copy

import torch
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from states import FitContext, BatchState
from configs import CoreComponents, HyperParameters
from metrics import Metrics
from data import ShortDataLoader, SizedIterable, set_dataloader_epoch
from engine_components.console_reporter import ConsoleReporter
from engine_components.logging import Logging
from engine_components.tqdm_reporter import TqdmReporter
from distributed_utils import is_distributed, is_main_process, barrier


class Engine:
    system_check_iteration_num: int = 10
    system_check_epoch_num: int = 5
    system_check_save_dir: str = "./.system_check/"

    def __init__(self, core_components: CoreComponents) -> None:
        self.core_components = core_components
        self.metrics = Metrics()
        self.console_reporter = ConsoleReporter()
        self.logging = Logging()
        self.tqdm_reporter = TqdmReporter()
        self.model = core_components.task_module

    def _prepare_model(self, device: str) -> None:
        # core_components.task_module（生のモジュール）は常に非ラップのまま保持し、
        # 保存/読込はそちらを使う。self.modelはDDP環境ならそれを丸ごとラップした別オブジェクト。
        task_module = self.core_components.task_module.to(device)
        if is_distributed():
            device_ids = [torch.device(device).index] if torch.device(device).type == "cuda" else None
            self.model = DistributedDataParallel(task_module, device_ids=device_ids)
        else:
            self.model = task_module

    def fit(self, fit_context: FitContext) -> None:
        # fit init
        self._prepare_model(fit_context.device)
        if is_main_process():
            os.makedirs(fit_context.save_dir, exist_ok=True)
        barrier()
        self.logging.save_parameters(self.core_components, fit_context)
        self.logging.load_checkpoint(self.core_components, fit_context)
        self.console_reporter.show_fit_detail(self.core_components, fit_context)

        for _ in range(fit_context.remaining_epoch):
            if not fit_context.on_fit:
                break
            self._epoch_loop(fit_context)

    def _epoch_loop(self, fit_context: FitContext) -> None:
        self.metrics.reset()
        # train loop
        set_dataloader_epoch(fit_context.train_dataloader, fit_context.global_state.current_epoch)
        self.model.train()
        self.tqdm_reporter.init_train_bar(fit_context)
        for batch_state in fit_context.train_dataloader:
            self._train_step(fit_context, batch_state)
        self.tqdm_reporter.reset_bar()
        fit_context.one_epoch()

        # val loop
        self.model.eval()
        self.tqdm_reporter.init_val_bar(fit_context)
        for batch_state in fit_context.val_dataloader:
            self._val_step(fit_context, batch_state)
        self.tqdm_reporter.reset_bar()

        # checkpoint save (chief processのみ書き込む)
        if is_main_process():
            self.logging.save_checkpoint(self.core_components, fit_context, self.metrics)

        # check result
        if fit_context.global_state.check_metric(self.metrics):
            if is_main_process():
                self.core_components.task_module.save(fit_context.save_dir)

    def _train_step(self, fit_context: FitContext, batch_state: BatchState) -> None:
        # init
        self.tqdm_reporter.on_train_batch(fit_context)
        self.core_components.optimizer.zero_grad(set_to_none=True)
        batch_state.to(fit_context.device)

        # step
        device_type = torch.device(fit_context.device).type
        dtype = getattr(torch, fit_context.hyper_parameters.amp) if fit_context.hyper_parameters.amp is not None else None
        with autocast(device_type=device_type, enabled=bool(fit_context.hyper_parameters.amp), dtype=dtype):
            batch_state = self.model(batch_state, fit_context, mode="train")

        if batch_state.loss is None:
            raise TypeError("lossが更新されていません。")
        if fit_context.grad_scaler:
            fit_context.grad_scaler.scale(batch_state.loss).backward()
            fit_context.grad_scaler.step(self.core_components.optimizer)
            fit_context.grad_scaler.update()
        else:
            batch_state.loss.backward()
            self.core_components.optimizer.step()

        if self.core_components.scheduler:
            self.core_components.scheduler.step()

        # post process
        batch_state.to("cpu")
        fit_context.one_step()
        self.metrics.update(batch_state, "train")
        self.tqdm_reporter.set_train_metrics(batch_state)

    def _val_step(self, fit_context: FitContext, batch_state: BatchState) -> None:
        # init
        self.tqdm_reporter.on_val_batch(fit_context)
        batch_state.to(fit_context.device)

        # step
        device_type = torch.device(fit_context.device).type
        dtype = getattr(torch, fit_context.hyper_parameters.amp) if fit_context.hyper_parameters.amp is not None else None
        with autocast(device_type=device_type, enabled=bool(fit_context.hyper_parameters.amp), dtype=dtype):
            batch_state = self.model(batch_state, fit_context, mode="val")

        # post process
        batch_state.to("cpu")
        self.metrics.update(batch_state, "val")
        self.tqdm_reporter.set_val_metrics(batch_state)

    def test(self, test_dataloader: SizedIterable[BatchState], hyper_parameters: HyperParameters, device: str) -> Metrics:
        self._prepare_model(device)
        self.model.eval()
        self.metrics.reset()

        # test loop
        self.tqdm_reporter.init_test_bar(test_dataloader)
        for batch_state in test_dataloader:
            self._test_step(hyper_parameters, device, batch_state)
        self.tqdm_reporter.reset_bar()

        return self.metrics

    def _test_step(self, hyper_parameters: HyperParameters, device: str, batch_state: BatchState) -> None:
        # init
        self.tqdm_reporter.on_test_batch()
        batch_state.to(device)

        # step
        device_type = torch.device(device).type
        dtype = getattr(torch, hyper_parameters.amp) if hyper_parameters.amp is not None else None
        with autocast(device_type=device_type, enabled=bool(hyper_parameters.amp), dtype=dtype):
            batch_state = self.model(batch_state, None, mode="test")

        # post process
        batch_state.to("cpu")
        self.metrics.update(batch_state, "test")
        self.tqdm_reporter.set_test_metrics(batch_state)

    def system_check(self, fit_context: FitContext) -> None:
        copy_fit_context = copy.deepcopy(fit_context)
        copy_core_components = copy.deepcopy(self.core_components)

        copy_fit_context.hyper_parameters = HyperParameters(
            max_epoch=self.system_check_epoch_num, batch_size=copy_fit_context.hyper_parameters.batch_size,
        )
        copy_fit_context.train_dataloader = ShortDataLoader(copy_fit_context.train_dataloader, self.system_check_iteration_num)
        copy_fit_context.val_dataloader = copy_fit_context.train_dataloader
        copy_fit_context.save_dir = self.system_check_save_dir

        try:
            self.fit(copy_fit_context)
            if is_main_process():
                print("精度が完璧に近づいたらシステムチェック成功.")
        finally:
            # fit()が例外で中断した場合でも、本番用のcore_componentsを学習前の状態に戻し、
            # チェック用ディレクトリを必ず片付ける
            self.core_components = copy_core_components
            barrier()
            if is_main_process() and os.path.exists(self.system_check_save_dir):
                shutil.rmtree(self.system_check_save_dir)
            barrier()
