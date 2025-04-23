
import os.path as osp

from ... import utils


def save_classification_trainer_params(self):
    # ================================
    # --------------------------------
    # --------------------------------
    for attr in (
            "configs_file_path",
            "device",
            "current_loss",
            "prev_loss",
            "max_epochs",
            "session",
            "class_labels",
            "checkpoints_path",
            "dataloaders",
            "model",
            "criterion",
            "optimizer",
            "scheduler",
            "evaluators",
    ):
        assert hasattr(self, attr), f"{self.__class__.__name__} does not have attribute `{attr}`"
    # --------------------------------
    # ================================

    utils.write_pickle_file(
        save_path=osp.join(self.checkpoints_path, "params.pk"),
        data={
            "configs_file_path": self.configs_file_path,
            "device": self.device,
            "current_loss": self.current_loss,
            "prev_loss": self.prev_loss,
            "max_epochs": self.max_epochs,
            "session": self.session,
            "class_labels": self.class_labels,
            "checkpoints_path": self.checkpoints_path,
            "dataloaders": self.dataloaders,
            "model": self.model,
            "criterion": self.criterion,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "evaluators": self.evaluators,
        }
    )