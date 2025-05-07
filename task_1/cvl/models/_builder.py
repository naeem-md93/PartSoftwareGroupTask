from torch import nn
from box import ConfigBox

from .try1._try1_model import Try1Model
from .try2._try2_model import Try2Model
from .try3._try3_model import Try3Model
from .try4._try4_model import Try4Model
from .try5._try5_model import Try5Model
from .try6._try6_model import Try6Model
from .try7._try7_model import Try7Model
from .try8._try8_model import Try8Model


BACKBONE_REGISTRY = {
    "Try1Model": Try1Model,
    "Try2Model": Try2Model,
    "Try3Model": Try3Model,
    "Try4Model": Try4Model,
    "Try5Model": Try5Model,
    "Try6Model": Try6Model,
    "Try7Model": Try7Model,
    "Try8Model": Try8Model,
}


def build_model(model_cfgs: ConfigBox) -> nn.Module:

    name = model_cfgs.pop("name")

    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"`{name}` is not a valid model name")

    model = BACKBONE_REGISTRY[name](**model_cfgs)

    return model
