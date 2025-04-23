from typing import Dict
from box import ConfigBox
from torch.utils.data import ConcatDataset

from .CIFAR10._dataset import CIFAR10Dataset


DATASET_REGISTRY = {
    "CIFAR10": CIFAR10Dataset,
}


def build_datasets(dataset_cfgs: ConfigBox) -> Dict[str, ConcatDataset]:

    tmp_datasets = {}
    datasets = {}

    for cfg in dataset_cfgs:
        name = cfg.pop("name")
        mode = cfg.pop("mode")

        if name not in DATASET_REGISTRY:
            print(f"Invalid dataset {name}!")
            continue

        tmp_datasets.setdefault(mode, []).append(DATASET_REGISTRY[name](mode=mode, **cfg))

    for ds_mode, ds_objs in tmp_datasets.items():
        datasets[ds_mode] = ConcatDataset(ds_objs)

    return datasets
