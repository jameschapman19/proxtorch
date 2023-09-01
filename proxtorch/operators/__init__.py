from .dummy import Dummy
from .elastic_net import ElasticNet
from .frobenius import Frobenius
from .fused_lasso import FusedLasso
from .graphnet import GraphNet3D, GraphNet2D
from .group_lasso import GroupLasso
from .huber import Huber
from .l1 import L1
from .l2 import L2
from .tracenorm import TraceNorm, NuclearNorm
from .tv_2d import TV_2D
from .tv_3d import TV_3D
from .tvl1_2d import TVL1_2D
from .tvl1_3d import TVL1_3D

__all__ = [
    "L1",
    "L2",
    "ElasticNet",
    "GroupLasso",
    "TV_2D",
    "TV_3D",
    "TVL1_2D",
    "TVL1_3D",
    "Frobenius",
    "TraceNorm",
    "NuclearNorm",
    "FusedLasso",
    "Huber",
    "Dummy",
    "GraphNet3D",
    "GraphNet2D",
]
