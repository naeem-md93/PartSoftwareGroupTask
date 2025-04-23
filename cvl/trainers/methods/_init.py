import torch
import shutil
import numpy as np
import os.path as osp
from torchinfo import summary

from ... import utils
from ...data.datasets import build_datasets
from ...data.utils import build_dataloaders
from ...models import build_model
from ...criterions import build_criterion
from ...optimizers import build_optimizer
from ...schedulers import build_scheduler
from ...evaluators import build_evaluator
from ...loggers import build_logger


def init_classification_trainer(self, **kwargs):

    # ================================
    for key in ("configs_file_path",):
        assert key in kwargs, f"{key} is not in kwargs"
    # --------------------------------
    self.configs_file_path = kwargs.pop("configs_file_path")
    # --------------------------------
    # --------------------------------
    # ================================

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.current_loss = np.inf
    self.prev_loss = np.inf

    cfgs = utils.read_yaml_config_file(self.configs_file_path)
    self.max_epochs = cfgs.pop('max_epochs', 20)
    self.session = cfgs.pop('session', "default")
    self.class_labels = cfgs.pop('class_labels', None)
    self.checkpoints_path = cfgs.pop('checkpoints_path', "checkpoints")
    self.checkpoints_path = osp.join(self.checkpoints_path, self.session)

    utils.mkdir(self.checkpoints_path)
    shutil.copy(src=self.configs_file_path, dst=osp.join(self.checkpoints_path, f"{self.session}_configs.yaml"))
    utils.set_seed(cfgs.pop('seed', None))

    self.dataloaders = build_dataloaders(build_datasets(cfgs.pop("datasets")), cfgs.pop("dataloaders"))

    self.model = build_model(cfgs.pop("model"))
    summary(
        model=self.model,
        input_size=(2, 3, 34, 34),
        col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"],
        depth=5,

    )

    self.criterion = build_criterion(cfgs.pop("criterion"))
    self.optimizer = build_optimizer(self.model.parameters(), cfgs.pop("optimizer"))
    self.scheduler = build_scheduler(self.optimizer, cfgs.pop("scheduler"))
    self.evaluators = {m: build_evaluator(m, self.class_labels, cfgs.get("evaluator")) for m in self.dataloaders.keys()}
    self.loggers = {m: build_logger(self.checkpoints_path, m, cfgs.get("logger")) for m in self.dataloaders.keys()}
    cfgs.pop("logger")

    self.loggers["train"].add_graph(self.model.to("cpu"), torch.tensor(np.random.random((2, 3, 34, 34)), dtype=torch.float))

    print(f"{cfgs=}")