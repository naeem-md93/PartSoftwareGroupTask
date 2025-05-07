from torch.optim import lr_scheduler


SCHEDULER_REGISTRY = {
    "StepLR": lr_scheduler.StepLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
}


def build_scheduler(optimizer, scheduler_configs):

    name = scheduler_configs.pop("name")

    if name not in SCHEDULER_REGISTRY:
        raise ValueError("Unknown scheduler: {}".format(name))

    return SCHEDULER_REGISTRY[name](optimizer, **scheduler_configs)