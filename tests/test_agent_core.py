# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for xmlx_vlm.agent_core primitives.
"""

import time
import pytest
from xmlx_vlm.agent_core import (
    ContextCompressor,
    SUMMARY_PREFIX,
    ThinkScrubber,
    ToolCallGuardrailConfig,
    ToolCallGuardrails,
    SubagentDelegator,
    SubagentTask,
    SubagentResult,
    BLOCKED_LEAF_TOOLS,
    estimate_tokens,
    estimate_message_tokens,
    prune_tool_outputs,
)


# ─── ThinkScrubber Tests ─────────────────────────────────────────────────────

def test_think_scrubber_basic():
    raw = "<think>Let me analyze the task.\nI should click button 12.</think>{\"tool\": \"click\", \"args\": {\"uid\": 12}}"
    clean, reasoning = ThinkScrubber.scrub(raw)
    assert clean == '{"tool": "click", "args": {"uid": 12}}'
    assert reasoning is not None
    assert "Let me analyze the task." in reasoning
    assert "I should click button 12." in reasoning


def test_think_scrubber_unclosed_tag():
    raw = "<think>Thinking about what to do next... no closing tag"
    clean, reasoning = ThinkScrubber.scrub(raw)
    assert clean == ""
    assert reasoning == "Thinking about what to do next... no closing tag"


def test_think_scrubber_no_tags():
    raw = "Just a direct response without tags."
    clean, reasoning = ThinkScrubber.scrub(raw)
    assert clean == raw
    assert reasoning is None


def test_think_scrubber_sanitize_messages():
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "<think>Processing user hello</think>Hello! How can I help?"},
    ]
    sanitized = ThinkScrubber.sanitize_messages(messages, keep_reasoning_in_metadata=True)
    assert len(sanitized) == 3
    assert sanitized[2]["content"] == "Hello! How can I help?"
    assert sanitized[2]["reasoning"] == "Processing user hello"


# ─── ToolCallGuardrails Tests ────────────────────────────────────────────────

def test_guardrails_exact_failure_threshold():
    config = ToolCallGuardrailConfig(
        warnings_enabled=True,
        hard_stop_enabled=True,
        exact_failure_warn_after=2,
        exact_failure_block_after=4,
    )
    guardrails = ToolCallGuardrails(config)

    # Call 1: Error (below warn)
    d1 = guardrails.observe_and_check("click", {"uid": 10}, "Error: Node not found", is_error=True)
    assert d1.action == "proceed"

    # Call 2: Exact same error (triggers warn)
    d2 = guardrails.observe_and_check("click", {"uid": 10}, "Error: Node not found", is_error=True)
    assert d2.action == "warn"
    assert "failed 2 times" in d2.reason

    # Call 3: Exact same error (still warn)
    d3 = guardrails.observe_and_check("click", {"uid": 10}, "Error: Node not found", is_error=True)
    assert d3.action == "warn"

    # Call 4: Exact same error (triggers block)
    d4 = guardrails.observe_and_check("click", {"uid": 10}, "Error: Node not found", is_error=True)
    assert d4.action == "block"
    assert d4.should_block is True
    assert "ACTION BLOCKED" in d4.synthetic_message


def test_guardrails_no_progress_detection():
    config = ToolCallGuardrailConfig(
        warnings_enabled=True,
        hard_stop_enabled=True,
        no_progress_warn_after=2,
        no_progress_block_after=3,
    )
    guardrails = ToolCallGuardrails(config)

    # Initial state
    guardrails.observe_and_check("navigate", {"url": "https://example.com"}, "OK", state_signature="https://example.com")

    # Mutating action 1 on same URL
    d1 = guardrails.observe_and_check("click", {"uid": 5}, "Clicked", state_signature="https://example.com")
    assert d1.action == "proceed"

    # Mutating action 2 on same URL -> warn
    d2 = guardrails.observe_and_check("click", {"uid": 6}, "Clicked", state_signature="https://example.com")
    assert d2.action == "warn"
    assert "unchanged" in d2.synthetic_message

    # Mutating action 3 on same URL -> block
    d3 = guardrails.observe_and_check("type_text", {"uid": 7, "text": "abc"}, "Typed", state_signature="https://example.com")
    assert d3.action == "block"
    assert "NO-PROGRESS BLOCKED" in d3.synthetic_message


def test_guardrails_abab_loop_detection():
    guardrails = ToolCallGuardrails()
    # A -> B -> A -> B
    guardrails.observe_and_check("scroll", {"direction": "down"}, "Scrolled down")
    guardrails.observe_and_check("scroll", {"direction": "up"}, "Scrolled up")
    guardrails.observe_and_check("scroll", {"direction": "down"}, "Scrolled down")
    d4 = guardrails.observe_and_check("scroll", {"direction": "up"}, "Scrolled up")

    assert d4.action == "warn"
    assert "LOOP DETECTED" in d4.synthetic_message


# ─── ContextCompressor Tests ─────────────────────────────────────────────────

def test_token_estimation():
    text = "Hello world! This is a test."
    tokens = estimate_tokens(text)
    assert tokens > 0

    image_msg = {"type": "image_url", "image_url": "data:image/png;base64,..."}
    img_tokens = estimate_tokens(image_msg)
    assert img_tokens == 1600


def test_prune_tool_outputs():
    long_result = "A" * 500
    messages = [
        {"role": "user", "content": "Search news"},
        {"role": "assistant", "content": "Calling snapshot"},
        {"role": "user", "content": f"Result: {long_result}"},
        {"role": "assistant", "content": "Calling click"},
        {"role": "user", "content": f"Result: {long_result}"},
        {"role": "assistant", "content": "Done"},
    ]
    # Protect tail count = 2 (so last 2 messages are protected)
    pruned = prune_tool_outputs(messages, protect_tail_count=2, max_pruned_len=50)
    # Message 2 (Result: ...) should be pruned
    assert "[Old tool output cleared to save context space]" in pruned[2]["content"]
    assert len(pruned[2]["content"]) < 300
    # Message 4 (Result: ...) should also be pruned because it's at index 4 (total 6, tail_threshold=4)
    # Message 5 is protected
    assert pruned[5]["content"] == "Done"


def test_context_compressor_compaction():
    compressor = ContextCompressor(
        max_context_tokens=200,
        compression_threshold=0.5,
        tail_token_budget=50,
    )

    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Task 1: do something long and comprehensive " * 5},
        {"role": "assistant", "content": "I am working on it " * 5},
        {"role": "user", "content": "Task 2: continue working " * 5},
        {"role": "assistant", "content": "I did step 2 " * 5},
        {"role": "user", "content": "Task 3: latest command"},
        {"role": "assistant", "content": "Final result step"},
    ]

    assert compressor.should_compress(messages) is True
    compacted, was_compressed = compressor.compress(messages)
    assert was_compressed is True
    assert len(compacted) < len(messages)
    # Ensure System prompt is preserved in Head
    assert compacted[0]["role"] == "system"
    # Ensure Summary message contains SUMMARY_PREFIX
    summary_msg = compacted[1]
    assert summary_msg["role"] == "user"
    assert SUMMARY_PREFIX in summary_msg["content"]
    # Ensure latest turn is preserved in Tail
    assert compacted[-1]["content"] == "Final result step"


# ─── SubagentDelegator Tests ─────────────────────────────────────────────────

def test_subagent_delegator_filter_tools():
    delegator = SubagentDelegator(agent_runner_fn=lambda t: None)
    all_tools = ["click", "snapshot", "delegate_task", "clarify", "read_file"]
    filtered_leaf = delegator.filter_available_tools(all_tools, role="leaf")
    assert "delegate_task" not in filtered_leaf
    assert "clarify" not in filtered_leaf
    assert "click" in filtered_leaf
    assert "snapshot" in filtered_leaf

    filtered_orch = delegator.filter_available_tools(all_tools, role="orchestrator")
    assert "delegate_task" in filtered_orch


def test_subagent_delegator_execution():
    def dummy_runner(task: SubagentTask) -> SubagentResult:
        return SubagentResult(
            task_id=task.task_id,
            success=True,
            summary=f"Finished goal: {task.goal}",
            total_steps=3,
            execution_time=0.05,
        )

    delegator = SubagentDelegator(agent_runner_fn=dummy_runner)
    task = SubagentTask(goal="Scrape product price", role="leaf")
    result = delegator.execute_single(task)

    assert result.success is True
    assert "Finished goal: Scrape product price" in result.summary
    assert result.total_steps == 3
    tool_resp = result.to_tool_result()
    assert "COMPLETED" in tool_resp
    assert task.task_id in tool_resp


def test_subagent_delegator_timeout():
    def hanging_runner(task: SubagentTask) -> SubagentResult:
        time.sleep(2.0)
        return SubagentResult(task_id=task.task_id, success=True, summary="Done", total_steps=1, execution_time=2.0)

    delegator = SubagentDelegator(agent_runner_fn=hanging_runner)
    task = SubagentTask(goal="Slow task", timeout_seconds=0.1)
    result = delegator.execute_single(task)

    assert result.success is False
    assert result.error == "TimeoutError"
    assert "timed out" in result.summary.lower()
