import torch
import os.path as osp


def save_classification_checkpoint(self, **kwargs) -> None:
    # ================================
    for key in ("epoch", "history"):
        assert key in kwargs, f"{key} is not in kwargs"
    # --------------------------------
    epoch = kwargs.pop("epoch")
    history = kwargs.pop("history")
    # --------------------------------
    for attr in ("checkpoints_path", "prev_loss", "current_loss", "model", "optimizer"):
        assert hasattr(self, attr), f"{self.__class__.__name__} does not have attribute `{attr}`"
    # --------------------------------
    # ================================

    sd = {
        "epoch": epoch,
        "history": history,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
    }

    torch.save(sd, osp.join(self.checkpoints_path, "last.pt"))

    if self.prev_loss > self.current_loss:
        torch.save(sd, osp.join(self.checkpoints_path, "best.pt"))
