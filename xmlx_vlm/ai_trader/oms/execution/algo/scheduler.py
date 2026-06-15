"""算法调度器."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderState
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.execution.algo.base import ExecutionAlgorithm, ParentOrder
from xmlx_vlm.ai_trader.oms.execution.algo.registry import get_algo
from xmlx_vlm.ai_trader.oms.routing.router import SmartOrderRouter
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class AlgoScheduler:
    """管理算法单生命周期."""

    def __init__(self, router: SmartOrderRouter):
        self._router = router
        self._algos: Dict[str, ExecutionAlgorithm] = {}
        self._parents: Dict[str, ParentOrder] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    @property
    def router(self) -> SmartOrderRouter:
        return self._router

    def start_algo(
        self,
        parent: ParentOrder,
        on_child_update: Optional[Any] = None,
    ) -> str:
        """启动算法，返回 algo_id."""
        algo_cls = get_algo(parent.algo_type)
        algo = algo_cls()
        self._algos[algo.algo_id] = algo
        self._parents[algo.algo_id] = parent
        parent.state = OrderState.SUBMITTED

        task = asyncio.create_task(
            self._run_algo(algo, parent, on_child_update),
            name=f"algo-{algo.algo_id}",
        )
        self._tasks[algo.algo_id] = task
        logger.info("Started algo %s (%s) for parent %s", algo.algo_id, parent.algo_type, parent.order_id)
        return algo.algo_id

    async def cancel_algo(self, algo_id: str) -> bool:
        """取消算法."""
        algo = self._algos.get(algo_id)
        if algo is None:
            return False
        algo.cancel()
        task = self._tasks.get(algo_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        parent = self._parents.get(algo_id)
        if parent is not None:
            parent.state = OrderState.CANCELLED
            parent.updated_at_ms = utc_now_ms()
        return True

    def get_parent(self, algo_id: str) -> Optional[ParentOrder]:
        return self._parents.get(algo_id)

    def get_algo(self, algo_id: str) -> Optional[ExecutionAlgorithm]:
        return self._algos.get(algo_id)

    def list_algos(self) -> Dict[str, Dict[str, Any]]:
        return {
            algo_id: {
                "algo_id": algo_id,
                "name": algo.name,
                "is_done": algo.is_done,
                "parent": parent.to_dict(),
            }
            for algo_id, (algo, parent) in self._iter_algos()
        }

    def _iter_algos(self):
        for algo_id, algo in self._algos.items():
            parent = self._parents.get(algo_id)
            if parent is not None:
                yield algo_id, (algo, parent)

    async def _run_algo(
        self,
        algo: ExecutionAlgorithm,
        parent: ParentOrder,
        on_child_update: Optional[Any] = None,
    ) -> None:
        try:
            await algo.start(parent, self._router, on_child_update)
        except asyncio.CancelledError:
            logger.info("Algo %s cancelled", algo.algo_id)
            if not parent.is_done():
                parent.state = OrderState.CANCELLED
                parent.updated_at_ms = utc_now_ms()
            raise
        except Exception as exc:
            logger.exception("Algo %s failed: %s", algo.algo_id, exc)
            parent.state = OrderState.REJECTED
            parent.reject_reason = str(exc)
            parent.updated_at_ms = utc_now_ms()

    def cleanup(self) -> None:
        """清理已完成任务."""
        done = [algo_id for algo_id, task in self._tasks.items() if task.done()]
        for algo_id in done:
            self._tasks.pop(algo_id, None)
