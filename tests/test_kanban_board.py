# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for xmlx_vlm.kanban (KanbanBoard, KanbanDispatcher).
"""

import os
import tempfile
import time
from pathlib import Path
import pytest

from xmlx_vlm.kanban import KanbanBoard, KanbanDispatcher, KanbanTask


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(prefix="kanban_test_", suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ─── KanbanBoard Tests ───────────────────────────────────────────────────────

def test_kanban_board_crud(temp_db_path):
    board = KanbanBoard(db_path=temp_db_path)

    # 1. Create tasks with different priorities
    t1 = board.create_task("Task low", "Description low", priority=1)
    t2 = board.create_task("Task high", "Description high", priority=5)

    # 2. List tasks (should be sorted by priority desc)
    tasks = board.list_tasks(status="todo")
    assert len(tasks) == 2
    assert tasks[0].id == t2.id
    assert tasks[1].id == t1.id

    # 3. Get single task
    fetched = board.get_task(t2.id)
    assert fetched is not None
    assert fetched.title == "Task high"
    assert fetched.priority == 5


def test_kanban_board_claim_and_complete(temp_db_path):
    board = KanbanBoard(db_path=temp_db_path)
    task = board.create_task("Scrape page", "Extract prices", assignee_profile="browser_agent", priority=3)

    # Claim with matching profile
    claimed = board.claim_task(assignee_profile="browser_agent", worker_id="worker-01")
    assert claimed is not None
    assert claimed.id == task.id
    assert claimed.status == "in_progress"
    assert claimed.worker_id == "worker-01"

    # Heartbeat
    hb_ok = board.heartbeat(task.id, worker_id="worker-01")
    assert hb_ok is True

    # Complete
    done_ok = board.complete_task(task.id, result_summary="Found 12 prices")
    assert done_ok is True

    completed_task = board.get_task(task.id)
    assert completed_task.status == "done"
    assert completed_task.result_summary == "Found 12 prices"


def test_kanban_board_failure_and_auto_block(temp_db_path):
    board = KanbanBoard(db_path=temp_db_path)
    task = board.create_task("Failing task", "Will fail repeatedly")

    # Fail 1
    board.fail_task(task.id, error="Network error", failure_limit=2)
    t_after_1 = board.get_task(task.id)
    assert t_after_1.failure_count == 1
    assert t_after_1.status == "todo"  # Retried

    # Fail 2 -> reaches failure_limit -> auto block
    board.fail_task(task.id, error="Fatal crash", failure_limit=2)
    t_after_2 = board.get_task(task.id)
    assert t_after_2.failure_count == 2
    assert t_after_2.status == "blocked"  # Blocked to prevent spin loop


def test_kanban_board_reclaim_stale(temp_db_path):
    board = KanbanBoard(db_path=temp_db_path)
    task = board.create_task("Hanging task", "Worker died", assignee_profile="default")
    board.claim_task("default", worker_id="dead-worker")

    # Manually backdate heartbeat
    one_hour_ago = time.time() - 3600
    with board._connect() as conn:
        conn.execute(
            "UPDATE kanban_tasks SET heartbeat_at = ? WHERE id = ?",
            (one_hour_ago, task.id),
        )
        conn.commit()

    # Reclaim
    reclaimed_count = board.reclaim_stale_tasks(stale_timeout_seconds=60.0)
    assert reclaimed_count == 1

    t_reclaimed = board.get_task(task.id)
    assert t_reclaimed.status == "todo"
    assert t_reclaimed.worker_id is None


# ─── KanbanDispatcher Tests ──────────────────────────────────────────────────

def test_kanban_dispatcher_flow(temp_db_path):
    board = KanbanBoard(db_path=temp_db_path)
    board.create_task("Process document", "Analyze PDF", assignee_profile="doc_worker")

    dispatcher = KanbanDispatcher(board=board, stale_timeout_seconds=10.0)

    # Register worker handler
    def mock_doc_worker(task: KanbanTask) -> str:
        return f"Successfully processed {task.title}"

    dispatcher.register_worker("doc_worker", mock_doc_worker)

    processed = dispatcher.dispatch_once()
    assert processed == 1

    tasks = board.list_tasks(status="done")
    assert len(tasks) == 1
    assert "Successfully processed" in tasks[0].result_summary
