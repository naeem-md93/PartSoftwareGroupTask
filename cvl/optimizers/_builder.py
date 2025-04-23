from torch import optim


OPTIMIZER_REGISTRY = {
    "SGD": optim.SGD
}


def build_optimizer(params, optimizer_config):

    name = optimizer_config.pop("name")

    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer {name} not found")

    return OPTIMIZER_REGISTRY[name](params=params, **optimizer_config)

