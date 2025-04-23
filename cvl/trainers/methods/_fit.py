import numpy as np

from ._train_one_epoch import train_one_classification_epoch
from ._test_one_epoch import test_one_classification_epoch
from ._save_checkpoint import save_classification_checkpoint


def fit_classification_trainer(self, **kwargs):

    # ================================
    for key in ("epoch", "modes"):
        assert key in kwargs, f"{key} is not in kwargs"
    # --------------------------------
    epoch = kwargs.pop("epoch")
    modes = kwargs.pop("modes")
    # --------------------------------
    for attr in (
        "current_loss",
        "prev_loss",
        "max_epochs",
        "device",
        "dataloaders",
        "model",
        "criterion",
        "optimizer",
        "scheduler",
        "evaluators",
        "loggers"
    ):
        assert hasattr(self, attr), f"{self.__class__.__name__} does not have attribute `{attr}`"
    # --------------------------------
    # ================================

    self.model = self.model.to(self.device)
    self.prev_loss = self.current_loss
    history = {}

    for ep in range(epoch, self.max_epochs + 1):
        if "train" in modes:
            self.model.train()
            self.evaluators["train"].reset()
            res = train_one_classification_epoch(
                ep,
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
            res = test_one_classification_epoch(
                ep,
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
            res = test_one_classification_epoch(
                ep,
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
        save_classification_checkpoint(self, epoch=ep, history=history)