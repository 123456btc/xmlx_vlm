"""执行算法注册表."""

from __future__ import annotations

from typing import Callable, Dict, Type

from xmlx_vlm.ai_trader.oms.execution.algo.base import ExecutionAlgorithm
from xmlx_vlm.ai_trader.oms.execution.algo.iceberg import IcebergAlgorithm
from xmlx_vlm.ai_trader.oms.execution.algo.liquidity_seek import LiquiditySeekAlgorithm
from xmlx_vlm.ai_trader.oms.execution.algo.twap import TWAPAlgorithm
from xmlx_vlm.ai_trader.oms.execution.algo.vwap import VWAPAlgorithm

ALGO_REGISTRY: Dict[str, Type[ExecutionAlgorithm]] = {
    "twap": TWAPAlgorithm,
    "vwap": VWAPAlgorithm,
    "iceberg": IcebergAlgorithm,
    "liquidity_seek": LiquiditySeekAlgorithm,
}


def get_algo(algo_type: str) -> Type[ExecutionAlgorithm]:
    algo_type = algo_type.lower()
    if algo_type not in ALGO_REGISTRY:
        raise ValueError(f"unsupported algo type: {algo_type}")
    return ALGO_REGISTRY[algo_type]
