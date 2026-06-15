from __future__ import annotations
from typing import NamedTuple
import mlx.core as mx

DEFAULT_TURBOQUANT_SEED = 0
_EPS = 1e-6
_POLAR_MAX_LEVELS = 4

class TurboQuantMSEState(NamedTuple):
    norms: mx.array
    indices: mx.array


class TurboQuantProdState(NamedTuple):
    norms: mx.array
    mse_indices: mx.array
    residual_norms: mx.array
    qjl_signs: mx.array


class TurboQuantPolarState(NamedTuple):
    radii: mx.array
    level_indices: tuple[mx.array, ...]


class TurboQuantPolarProdState(NamedTuple):
    norms: mx.array
    polar_state: TurboQuantPolarState
    residual_norms: mx.array
    qjl_signs: mx.array


class TurboQuantSplitState(NamedTuple):
    low: object
    high: object


