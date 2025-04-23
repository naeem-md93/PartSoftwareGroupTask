from torch import nn


CRITERION_REGISTRY = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}


def build_criterion(criterion_configs):
    name = criterion_configs.pop("name")

    if name not in CRITERION_REGISTRY:
        raise KeyError(f"Invalid criterion {name}!")

    return CRITERION_REGISTRY[name](**criterion_configs)
