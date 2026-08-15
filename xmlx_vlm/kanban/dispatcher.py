# SPDX-License-Identifier: Apache-2.0
"""
Kanban Dispatcher -- Long-running task orchestrator for specialist agents.

Polls the KanbanBoard, dispatches ready tasks to matching worker profiles,
maintains heartbeats, and reclaims dead/stale workers.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from xmlx_vlm.kanban.board import KanbanBoard, KanbanTask

logger = logging.getLogger(__name__)

WorkerRunnerFn = Callable[[KanbanTask], str]


class KanbanDispatcher:
    """
    Orchestrates background dispatching of Kanban tasks to specialized worker functions.
    """

    def __init__(
        self,
        board: Optional[KanbanBoard] = None,
        poll_interval: float = 2.0,
        stale_timeout_seconds: float = 120.0,
        failure_limit: int = 2,
    ):
        self.board = board or KanbanBoard()
        self.poll_interval = poll_interval
        self.stale_timeout_seconds = stale_timeout_seconds
        self.failure_limit = failure_limit
        self.workers: Dict[str, WorkerRunnerFn] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_worker(self, profile_name: str, runner_fn: WorkerRunnerFn) -> None:
        """Register a worker runner function for a specific profile (or 'default')."""
        self.workers[profile_name] = runner_fn
        logger.info("Registered kanban worker for profile: %s", profile_name)

    def dispatch_once(self) -> int:
        """
        Execute one dispatch tick:
        1. Reclaim stale tasks.
        2. Attempt claiming and executing tasks across all registered profiles.
        Returns the number of tasks processed.
        """
        # Step 1: Reclaim expired tasks
        reclaimed = self.board.reclaim_stale_tasks(self.stale_timeout_seconds)
        if reclaimed > 0:
            logger.info("Reclaimed %d stale in-progress tasks back to todo", reclaimed)

        processed_count = 0

        # Step 2: Attempt task claim for each registered profile
        for profile, runner in list(self.workers.items()):
            worker_id = f"{profile}-{uuid.uuid4().hex[:6]}"
            task = self.board.claim_task(assignee_profile=profile, worker_id=worker_id)
            if not task:
                continue

            logger.info("Worker [%s] claimed task [%s]: %s", worker_id, task.id, task.title)
            processed_count += 1

            # Heartbeat updater thread during execution
            stop_heartbeat = threading.Event()

            def _heartbeat_loop():
                while not stop_heartbeat.wait(timeout=max(1.0, self.stale_timeout_seconds / 3)):
                    self.board.heartbeat(task.id, worker_id)

            hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            hb_thread.start()

            try:
                result_summary = runner(task)
                self.board.complete_task(task.id, result_summary=str(result_summary))
                logger.info("Task [%s] completed successfully by %s", task.id, worker_id)
            except Exception as e:
                logger.error("Task [%s] failed under worker %s: %s", task.id, worker_id, e)
                self.board.fail_task(task.id, error=str(e), failure_limit=self.failure_limit)
            finally:
                stop_heartbeat.set()
                hb_thread.join(timeout=2.0)

        return processed_count

    def start(self) -> None:
        """Start the dispatcher background loop."""
        if self._running:
            return
        self._running = True

        def _loop():
            logger.info("Kanban dispatcher loop started (poll_interval=%.1fs)", self.poll_interval)
            while self._running:
                try:
                    self.dispatch_once()
                except Exception as e:
                    logger.error("Error in kanban dispatch loop: %s", e)
                time.sleep(self.poll_interval)
            logger.info("Kanban dispatcher loop stopped")

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the dispatcher background loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
