from .l1 import L1Prox
from .l2 import L2Prox
from .elastic_net import ElasticNetProx
from .group_lasso import GroupLassoProx
from .non_negative import NonNegativeProx
from .l0ball import L0Ball
from .l1ball import L1Ball
from .tv_2d import TV_2DProx
from .tv_3d import TV_3DProx
from .tvl1_2d import TVL1_2DProx
from .tvl1_3d import TVL1_3DProx
from .frobenius import FrobeniusProx
from .tracenorm import TraceNormProx
from .fused_lasso import FusedLassoProx
from .huber import HuberProx

__all__ = [
    "L1Prox",
    "L2Prox",
    "ElasticNetProx",
    "GroupLassoProx",
    "NonNegativeProx",
    "L0Ball",
    "L1Ball",
    "TV_2DProx",
    "TV_3DProx",
    "TVL1_2DProx",
    "TVL1_3DProx",
    "FrobeniusProx",
    "TraceNormProx",
    "FusedLassoProx",
    "HuberProx",
]
