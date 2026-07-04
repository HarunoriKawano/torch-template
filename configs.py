from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict
import torch
from torch.optim import Optimizer

from task_module import TaskModule

class HyperParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_epoch: int
    batch_size: int
    device: str
    cpu_num_works: int
    amp: Optional[Literal["float16", "bfloat16"]] = None
    max_patient_num: Optional[int] = None

class CoreComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_module: TaskModule
    optimizer: Optimizer

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.task_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"Core components saved to {path}")

    def load(self, path: str, device: torch.device):
        checkpoint = torch.load(path, map_location=device)

        self.task_module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Core components loaded from {path}")
