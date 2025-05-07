import gc

import matplotlib.pyplot as plt
import torch
import shutil
import numpy as np
import os.path as osp
from rich.live import Live
from torchinfo import summary
from torchvision.transforms import v2
import torch.nn.functional as F

from .. import utils
from ..data.datasets import build_datasets
from ..data.utils import build_dataloaders
from ..models import build_model
from ..criterions import build_criterion
from ..optimizers import build_optimizer
from ..schedulers import build_scheduler
from ..evaluators import build_evaluator
from ..loggers import build_logger


class ClassificationTrainer:
    def __init__(self, configs_file_path: str) -> None:
        self.configs_file_path = configs_file_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_loss = np.inf
        self.prev_loss = np.inf
        self.batch_trans = v2.RandomChoice(transforms=[
            v2.RandomHorizontalFlip(p=0),
            v2.RandomErasing(p=1, scale=(0.05, 0.33), ratio=(0.5, 2.0), value=(0.49139968, 0.48215841, 0.44653091)),
            v2.MixUp(alpha=1.0, num_classes=10),
            v2.CutMix(alpha=1.0, num_classes=10)
        ], p=None)

        cfgs = utils.read_yaml_config_file(self.configs_file_path)
        self.max_epochs = cfgs.pop('max_epochs', 20)
        self.session = cfgs.pop('session', "default")
        self.class_labels = cfgs.pop('class_labels', None)
        self.visualizer_target_layers = cfgs.pop('visualizer_target_layers', [])
        self.checkpoints_path = cfgs.pop('checkpoints_path', "checkpoints")
        self.checkpoints_path = osp.join(self.checkpoints_path, self.session)

        utils.mkdir(self.checkpoints_path)
        shutil.copy(src=self.configs_file_path, dst=osp.join(self.checkpoints_path, f"{self.session}_configs.yaml"))
        utils.set_seed(cfgs.pop('seed', None))

        self.dataloaders = build_dataloaders(build_datasets(cfgs.pop("datasets")), cfgs.pop("dataloaders"))

        self.model = build_model(cfgs.pop("model"))
        self.criterion = build_criterion(cfgs.pop("criterion"))
        self.optimizer = build_optimizer(self.model.parameters(), cfgs.pop("optimizer"))
        self.scheduler = build_scheduler(self.optimizer, cfgs.pop("scheduler"))
        self.evaluators = {m: build_evaluator(m, self.class_labels, cfgs.get("evaluator")) for m in self.dataloaders.keys()}
        cfgs.pop("evaluator")

        self.loggers = {m: build_logger(self.checkpoints_path, m, cfgs.get("logger")) for m in self.dataloaders.keys()}
        cfgs.pop("logger")

        summary(
            model=self.model,
            input_size=(2, 3, 32, 32),
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable"
            ],
            depth=5,

        )
        self.loggers["train"].add_graph(self.model.to("cpu"), torch.tensor(np.random.random((2, 3, 32, 32)), dtype=torch.float))

    def save_params(self):
        utils.write_pickle_file(
            save_path=osp.join(self.checkpoints_path, "params.pk"),
            data={
                "configs_file_path": self.configs_file_path,
                "current_loss": self.current_loss,
                "prev_loss": self.prev_loss,
                "batch_trans": self.batch_trans,
                "max_epochs": self.max_epochs,
                "session": self.session,
                "class_labels": self.class_labels,
                "visualizer_target_layers": self.visualizer_target_layers,
                "checkpoints_path": self.checkpoints_path,
                "dataloaders": self.dataloaders,
                "model": self.model,
                "criterion": self.criterion,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "evaluators": self.evaluators,
            }
        )

    def save_checkpoint(self, epoch: int, history: dict) -> None:

        sd = {
            "epoch": epoch,
            "history": history,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(sd, osp.join(self.checkpoints_path, "last.pt"))

        if self.prev_loss > self.current_loss:
            torch.save(sd, osp.join(self.checkpoints_path, "best.pt"))

    def fit(self, epoch: int = 1, modes: tuple[str] = ("train", "validation")) -> None:
        self.model = self.model.to(self.device)
        self.prev_loss = self.current_loss
        history = {}

        for ep in range(epoch, self.max_epochs + 1):
            if "train" in modes:
                self.model.train()
                self.evaluators["train"].reset()
                res = self.train_one_epoch(
                    ep,
                    self.max_epochs,
                    "train",
                    self.device,
                    self.dataloaders["train"],
                    self.model,
                    self.criterion,
                    self.optimizer,
                    self.evaluators["train"],
                    self.loggers["train"],
                )
                history.setdefault("train", []).append(res)

            if "validation" in modes:
                self.model.eval()
                self.evaluators["validation"].reset()
                res = self.test_one_epoch(
                    ep,
                    self.max_epochs,
                    "validation",
                    self.device,
                    self.dataloaders["validation"],
                    self.model,
                    self.criterion,
                    self.evaluators["validation"],
                    self.loggers["validation"],
                )
                history.setdefault("validation", []).append(res)

                self.current_loss = np.mean(res["L"])

            if "test" in modes:
                self.model.eval()
                self.evaluators["test"].reset()
                res = self.test_one_epoch(
                    ep,
                    self.max_epochs,
                    "test",
                    self.device,
                    self.dataloaders["test"],
                    self.model,
                    self.criterion,
                    self.evaluators["test"],
                    self.loggers["test"],
                )
                history.setdefault("test", []).append(res)

                self.current_loss = np.mean(res["L"])

            self.scheduler.step()
            self.save_checkpoint(ep, history)

    def fit_test(self):
        self.model = self.model.to(self.device)
        self.prev_loss = self.current_loss
        history = {}
        self.model.eval()
        self.evaluators["test"].reset()
        res = self.test_one_epoch(
            1,
            1,
            "test",
            self.device,
            self.dataloaders["test"],
            self.model,
            self.criterion,
            self.evaluators["test"],
            self.loggers["test"],
        )
        history.setdefault("test", []).append(res)

        self.current_loss = np.mean(res["L"])

    def train_one_epoch(
        self,
        epoch: int,
        max_epochs: int,
        mode: str,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        evaluator,
        logger
    ):
        H = {"L": [], "A": []}
        cm_title = f"\nM: {mode} | E: {epoch}/{max_epochs} |" + " B: {0}/" + str(
            len(dataloader)) + " | L: {1} | A: {2} | LR: {3} |"

        with Live(evaluator.build_cm_table(title=cm_title.format(0, 0, 0, 0)), refresh_per_second=1) as live:
            for bn, data in enumerate(dataloader):

                images = data["image"].to(device)
                labels = data["label"].to(device)

                images, labels = self.batch_trans(images, labels)

                logits = model(images)

                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                gc.collect()
                torch.cuda.empty_cache()

                evaluator.update(labels, logits)

                acc = evaluator.get_accuracy()
                H["L"].append(loss.item())
                H["A"].append(acc)

                logger.add_scalar(f"{mode}/batch/loss", loss.item())
                logger.add_scalar(f"{mode}/batch/accuracy", acc)

                live.update(evaluator.build_cm_table(title=cm_title.format(
                    bn + 1,
                    round(np.mean(H['L']), 2),
                    round(np.mean(H['A']), 2),
                    round(optimizer.param_groups[0]['lr'], 4)
                )))

            H["CM"] = evaluator.get_confusion_matrix()

            logger.add_scalar(f"{mode}/epoch/loss", np.mean(H["L"]))
            logger.add_scalar(f"{mode}/epoch/accuracy", np.mean(H["A"]))

            return H

    def test_one_epoch(
        self,
        epoch: int,
        max_epochs: int,
        mode: str,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        evaluator,
        logger
    ):
        H = {"L": [], "A": []}
        cm_title = f"\nM: {mode} | E: {epoch}/{max_epochs} |" + " B: {0}/" + str(
            len(dataloader)) + " | L: {1} | A: {2} |"


        with Live(evaluator.build_cm_table(title=cm_title.format(0, 0, 0)), refresh_per_second=1) as live:
            for bn, data in enumerate(dataloader):

                images = data["image"].to(device)
                labels = data["label"].to(device)

                with torch.no_grad():
                    logits = model(images)

                loss = criterion(logits, labels)

                gc.collect()
                torch.cuda.empty_cache()

                evaluator.update(labels, logits)

                acc = evaluator.get_accuracy()
                H["L"].append(loss.item())
                H["A"].append(acc)

                logger.add_scalar(f"{mode}/batch/loss", loss.item())
                logger.add_scalar(f"{mode}/batch/accuracy", acc)

                live.update(evaluator.build_cm_table(title=cm_title.format(
                    bn + 1,
                    round(np.mean(H['L']), 2),
                    round(np.mean(H['A']), 2),
                )))

            H["CM"] = evaluator.get_confusion_matrix()

            logger.add_scalar(f"{mode}/epoch/loss", np.mean(H["L"]))
            logger.add_scalar(f"{mode}/epoch/accuracy", np.mean(H["A"]))

            return H

    def visualize(self, checkpoints_path: str) -> None:

        params = utils.read_pickle_file(osp.join(checkpoints_path, "params.pk"))
        dataloader = params["dataloaders"]["validation"]
        model = params["model"]
        visualizer_target_layers = params["visualizer_target_layers"]
        class_labels = params["class_labels"]
        checkpoints_path = params["checkpoints_path"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sd = torch.load(osp.join(checkpoints_path, "best.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict=sd["model_state_dict"], strict=True)

        vis = utils.Visualizer(dataloader, model, visualizer_target_layers, class_labels, checkpoints_path, device)
        vis.save_maps()
