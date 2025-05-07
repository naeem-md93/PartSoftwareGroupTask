from box import ConfigBox

from ._tensorboard import TensorboardLogger


LOGGER_REGISTRY = {
    "Tensorboard": TensorboardLogger,
}


def build_logger(checkpoints_path: str, mode: str, logger_configs: ConfigBox):

    name = logger_configs.get("name")

    if name not in LOGGER_REGISTRY:
        raise KeyError("Unknown logger name: {}".format(name))

    return LOGGER_REGISTRY[name](checkpoints_path, mode, **logger_configs)
