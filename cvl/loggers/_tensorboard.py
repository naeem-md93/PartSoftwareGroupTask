import os
import torch
from torch.utils.tensorboard import SummaryWriter

from .. import utils

class TensorboardLogger:
    def __init__(self, checkpoints_path: str, mode: str, name: str) -> None:
        log_dir = os.path.join(checkpoints_path, "tensorboard", mode)
        utils.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_steps = {}

    def add_scalar(self, tag: str, scalar: torch.Tensor) -> None:
        self.global_steps[tag] = self.global_steps.get(tag, 0) + 1

        self.writer.add_scalar(
            tag=tag,
            scalar_value=scalar,
            global_step=self.global_steps[tag]
        )

    def add_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor) -> None:
        self.writer.add_graph(model, input_to_model)

