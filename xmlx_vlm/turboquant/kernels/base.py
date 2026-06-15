from __future__ import annotations
import mlx.core as mx

def _metal_available() -> bool:
    return hasattr(mx, "metal") and mx.metal.is_available()

