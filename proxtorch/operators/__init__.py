from .dummy import DummyProx
from .elastic_net import ElasticNetProx
from .frobenius import FrobeniusProx
from .fused_lasso import FusedLassoProx
from .graphnet import GraphNet3DProx, GraphNet2DProx
from .group_lasso import GroupLassoProx
from .huber import HuberProx
from .l1 import L1Prox
from .l2 import L2Prox
from .tracenorm import TraceNormProx, NuclearNormProx
from .tv_2d import TV_2DProx
from .tv_3d import TV_3DProx
from .tvl1_2d import TVL1_2DProx
from .tvl1_3d import TVL1_3DProx

__all__ = [
    "L1Prox",
    "L2Prox",
    "ElasticNetProx",
    "GroupLassoProx",
    "TV_2DProx",
    "TV_3DProx",
    "TVL1_2DProx",
    "TVL1_3DProx",
    "FrobeniusProx",
    "TraceNormProx",
    "NuclearNormProx",
    "FusedLassoProx",
    "HuberProx",
    "DummyProx",
    "GraphNet3DProx",
    "GraphNet2DProx",
]
