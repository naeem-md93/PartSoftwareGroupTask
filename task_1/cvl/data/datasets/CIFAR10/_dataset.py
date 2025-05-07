from typing import Optional, Tuple, Dict
import torch
from box import ConfigBox
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10

from . import _utils
from ...transforms import build_transforms


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, mode: str, task: str, dataset_path: str, transforms: Optional[ConfigBox] = None) -> None:
        super(CIFAR10Dataset, self).__init__()

        if mode in ("train", "validation"):
            _ds = CIFAR10(root=dataset_path, train=True, download=True)
        elif mode == "test":
            _ds = CIFAR10(root=dataset_path, train=False, download=True)
        else:
            raise ValueError(f"Invalid mode `{mode}`")

        self.class_to_idx = _ds.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.transforms = v2.Compose(build_transforms(transforms))
        self.data = _utils.get_data(_ds, mode)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        img, lbl = self.data[idx]

        label = torch.zeros(size=(len(self.class_to_idx), ), dtype=torch.float32)
        label[lbl] = 1

        data = {
            "image": img,
            "label": label,
        }
        if self.transforms is not None:
            data = self.transforms(data)

        return data



