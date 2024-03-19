from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType, Tuple, Any


class Activations(Enum):
    relu = auto()
    relu6 = auto()
    lrelu = auto()
    tanh = auto()
    sigmoid = auto()
    elu = auto()


class Operations(Enum):
    conv = auto()
    bn = auto()
    max_pool = auto()
    avg_pool = auto()
    dropout = auto()
    upsample = auto()
    activation = auto()


class WeightInit(Enum):
    kaiming = auto()
    xavier = auto()
    zeros = auto()
    ones = auto()
    default = auto()  # will use the default initialization of pytorch


ParameterizedOps = NewType("ParameterizedOps", Tuple[Operations, dict[str, Any]])
