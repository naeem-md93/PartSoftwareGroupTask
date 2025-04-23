from typing import List, Tuple
import random
from PIL import Image
from torchvision.datasets import CIFAR10


def get_data(cifar_dataset: CIFAR10, mode: str) -> List[Tuple[Image, int]]:
    assert mode in ("train", "validation", "test"), f"Invalid mode {mode}"

    if mode == "test":
        return [x for x in cifar_dataset]

    category_images = {}
    for (img, lbl) in cifar_dataset:
        category_images.setdefault(lbl, []).append((img, lbl))

    for k in category_images.keys():
        random.shuffle(category_images[k])

    train_data = []
    valid_data = []

    for cat, imgs in category_images.items():
        valid_data += imgs[:500]
        train_data += imgs[500:]

    random.shuffle(train_data)
    random.shuffle(valid_data)

    if mode == "train":
        return train_data

    return valid_data
