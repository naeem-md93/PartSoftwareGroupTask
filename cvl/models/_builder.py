from torch import nn
from box import ConfigBox

from .try1._try1_model import Try1Model
from .try2._try2_model import Try2Model


BACKBONE_REGISTRY = {
    "Try1Model": Try1Model,
    "Try2Model": Try2Model,
}


def build_model(model_cfgs: ConfigBox) -> nn.Module:

    name = model_cfgs.pop("name")

    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"`{name}` is not a valid model name")

    model = BACKBONE_REGISTRY[name](**model_cfgs)

    return model
