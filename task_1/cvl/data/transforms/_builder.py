from typing import Dict, List, Tuple, Optional, Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


TODATATYPE_KWARGS = {
    "float64": torch.float64,
    "float32": torch.float32,

    "int32": torch.int32,
    "int64": torch.int64,

    "uint8": torch.uint8,
    "uint16": torch.uint16,

    "long": torch.long,

    "image": tv_tensors.Image,
    "label": "label"
}


TRANSFORMS_REGISTRY = {
    # Geometric
    "Pad": v2.Pad,
    "Resize": v2.Resize,
    "CenterCrop": v2.CenterCrop,
    "FiveCrop": v2.FiveCrop,
    "RandomPerspective": v2.RandomPerspective,
    "RandomRotation": v2.RandomRotation,
    "RandomAffine": v2.RandomAffine,
    "ElasticTransform": v2.ElasticTransform,
    "RandomCrop": v2.RandomCrop,
    "RandomResizedCrop": v2.RandomResizedCrop,
    "RandomHorizontalFlip": v2.RandomHorizontalFlip,
    "RandomVerticalFlip": v2.RandomVerticalFlip,
    "RandomErasing": v2.RandomErasing,

    # Photometric
    "Grayscale": v2.Grayscale,
    "ColorJitter": v2.ColorJitter,
    "GaussianBlur": v2.GaussianBlur,
    "RandomInvert": v2.RandomInvert,
    "RandomPosterize": v2.RandomPosterize,
    "RandomSolarize": v2.RandomSolarize,
    "RandomAdjustSharpness": v2.RandomAdjustSharpness,
    "RandomAutocontrast": v2.RandomAutocontrast,
    "RandomEqualize": v2.RandomEqualize,
    "JPEG": v2.JPEG,

    # General
    "AutoAugment": v2.AutoAugment,
    "MixUp": v2.MixUp,
    "CutMix": v2.CutMix,
    "AugMix": v2.AugMix,
    "RandomChoice": v2.RandomChoice,
    "ClampBoundingBoxes": v2.ClampBoundingBoxes,
    "SanitizeBoundingBoxes": v2.SanitizeBoundingBoxes,
    "ToImage": v2.ToImage,
    "ToDtype": v2.ToDtype,
    "Normalize": v2.Normalize,

}


AutoAugment_POLICIES = {
    "CIFAR10": v2.AutoAugmentPolicy.CIFAR10,
}

def build_transforms(tran_list: Optional[List[dict[str, Any]]] = None) -> Optional[List[v2.Transform]]:

    if (tran_list is None) or (len(tran_list) == 0):
        return None

    transforms = []
    for cfg in tran_list:
        if (cfg is None) or (len(cfg) == 0):
            print(f"WARNING: no transform configs defined in {cfg}")
            continue

        name = cfg.pop("name")
        if name not in TRANSFORMS_REGISTRY:
            print(f"WARNING: Invalid transforms {name}: {cfg}")
            continue

        if name == "AutoAugment":
            cfg["policy"] = AutoAugment_POLICIES[cfg["policy"]]

        elif name == "RandomChoice":
            transforms.append(
                v2.RandomChoice(
                    transforms=build_transforms(cfg.pop("transforms")),
                    p=cfg.pop("p", None)
                )
            )

        elif name == "ToDtype":
            new_cfg = {"dtype": {}, "scale": cfg.pop("scale", False)}

            for k, v in cfg["dtype"].items():
                new_cfg["dtype"][TODATATYPE_KWARGS[k]] = TODATATYPE_KWARGS[v]
            transforms.append(v2.ToDtype(**new_cfg))

        else:
            transforms.append(TRANSFORMS_REGISTRY[name](**cfg))

    return transforms
