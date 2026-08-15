# SPDX-License-Identifier: Apache-2.0
"""
Subagent Delegation -- Isolated child agent execution and budget management.

Provides subtask dispatching with:
1. Complete context isolation (child conversation does not pollute parent)
2. Toolset restriction (Leaf subagents cannot recurse or trigger side effects)
3. Non-interactive deadlock protection (worker threads never block on stdin)
4. Structured summary return (only the final conclusion is injected into parent)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Tools strictly blocked for leaf subagents to avoid recursion and side effects
BLOCKED_LEAF_TOOLS: Set[str] = {
    "delegate_task",
    "clarify",
    "send_message",
    "memory_write",
}


@dataclass
class SubagentTask:
    """Definition of a delegated subtask."""

    goal: str
    context: Optional[str] = None
    role: str = "leaf"  # "leaf" or "orchestrator"
    max_steps: int = 15
    timeout_seconds: float = 120.0
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class SubagentResult:
    """Summary result returned to the parent agent."""

    task_id: str
    success: bool
    summary: str
    total_steps: int
    execution_time: float
    error: Optional[str] = None

    def to_tool_result(self) -> str:
        """Format as a clean tool execution response for the parent LLM."""
        status = "COMPLETED" if self.success else "FAILED"
        return json.dumps(
            {
                "status": status,
                "task_id": self.task_id,
                "summary": self.summary,
                "steps_taken": self.total_steps,
                "execution_seconds": round(self.execution_time, 2),
                "error": self.error,
            },
            ensure_ascii=False,
            indent=2,
        )


class SubagentDelegator:
    """
    Manages spawning, executing, and aggregating subagent tasks.
    """

    def __init__(
        self,
        agent_runner_fn: Callable[[SubagentTask], SubagentResult],
        max_concurrent_tasks: int = 3,
        auto_approve_dangerous: bool = False,
    ):
        self.agent_runner_fn = agent_runner_fn
        self.max_concurrent_tasks = max_concurrent_tasks
        self.auto_approve_dangerous = auto_approve_dangerous

    def filter_available_tools(self, tool_names: List[str], role: str = "leaf") -> List[str]:
        """Filter out blocked tools if the subagent is a leaf worker."""
        if role == "leaf":
            return [t for t in tool_names if t not in BLOCKED_LEAF_TOOLS]
        return tool_names

    def execute_single(self, task: SubagentTask) -> SubagentResult:
        """Run a single subtask with timeout protection."""
        t0 = time.time()
        logger.info("Executing delegated subtask [%s]: %s (role=%s)", task.task_id, task.goal[:60], task.role)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.agent_runner_fn, task)
            try:
                result = future.result(timeout=task.timeout_seconds)
                return result
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - t0
                logger.warning("Subtask [%s] timed out after %.1fs", task.task_id, elapsed)
                return SubagentResult(
                    task_id=task.task_id,
                    success=False,
                    summary=f"Subtask timed out after {task.timeout_seconds} seconds.",
                    total_steps=task.max_steps,
                    execution_time=elapsed,
                    error="TimeoutError",
                )
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("Subtask [%s] encountered unhandled exception: %s", task.task_id, e)
                return SubagentResult(
                    task_id=task.task_id,
                    success=False,
                    summary=f"Subtask failed with error: {str(e)}",
                    total_steps=0,
                    execution_time=elapsed,
                    error=str(e),
                )

    def execute_batch(self, tasks: List[SubagentTask]) -> List[SubagentResult]:
        """Run multiple subtasks concurrently, respecting concurrency limits."""
        if not tasks:
            return []

        results: List[SubagentResult] = []
        max_workers = min(len(tasks), self.max_concurrent_tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.execute_single, task): task for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    task = future_to_task[future]
                    results.append(
                        SubagentResult(
                            task_id=task.task_id,
                            success=False,
                            summary=f"Batch execution failed: {e}",
                            total_steps=0,
                            execution_time=0.0,
                            error=str(e),
                        )
                    )
        return results
