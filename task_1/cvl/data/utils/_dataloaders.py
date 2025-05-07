from typing import Dict
from box import ConfigBox
from torch.utils.data import DataLoader, ConcatDataset


def build_dataloaders(datasets: Dict[str, ConcatDataset], dataloader_cfgs: ConfigBox) -> Dict[str, DataLoader]:

    dataloaders = {}

    for mode, cfgs in dataloader_cfgs.items():
        if mode not in datasets:
            print(f"Warning: dataset mode `{mode}` not found, skipping")
            continue

        dataloaders[mode] = DataLoader(dataset=datasets[mode], **cfgs)

    return dataloaders
