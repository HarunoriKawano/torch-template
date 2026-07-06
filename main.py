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
        engine.test()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
