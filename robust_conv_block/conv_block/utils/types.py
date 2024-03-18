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


class WeightInit(Enum):
    kaiming = auto()
    xavier = auto()
    normal = auto()
    uniform = auto()
    zeros = auto()
    ones = auto


ParameterizedOps = NewType("ParameterizedOps", Tuple[Operations, dict[str, Any]])
