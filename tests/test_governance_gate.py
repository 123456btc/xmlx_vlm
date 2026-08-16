# SPDX-License-Identifier: Apache-2.0
"""
Tests for Governance Gate and ClockProvider Determinism.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from xmlx_vlm.ai_trader.oms.utils.clock import IClock, RealtimeClock, VirtualClock, get_clock, set_clock, reset_clock
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms, utc_now_iso


def test_clock_provider_virtual_stepping():
    """Test that VirtualClock provides deterministic timestamps and monotonic offsets."""
    start_ts = 1700000000000
    vclock = VirtualClock(start_ms=start_ts)
    set_clock(vclock)

    try:
        assert utc_now_ms() == start_ts
        assert get_clock().monotonic() == 0.0
        
        # Advance clock by 5000 ms (5 seconds)
        vclock.advance_ms(5000)
        assert utc_now_ms() == start_ts + 5000
        assert get_clock().monotonic() == 5.0
        
        # Advance by another 1000 ms
        vclock.advance_ms(1000)
        assert utc_now_ms() == start_ts + 6000
        assert get_clock().monotonic() == 6.0
    finally:
        reset_clock()


def test_governance_constitution_files_exist():
    """Test that all required .governance files are present and well-formed."""
    root_dir = Path(__file__).resolve().parent.parent
    gov_dir = root_dir / ".governance"
    
    assert gov_dir.exists() and gov_dir.is_dir()
    
    required_files = [
        "GOVERNANCE_PRIMACY.md",
        "ENGINEERING_CONSTITUTION.md",
        "GOVERNANCE_MAP.md",
        "AI_GOVERNANCE_OPERATING_MODEL.md",
        "PROJECT_BUGS_POSTMORTEM.md",
        "scripts/ci_gate.py",
        "scripts/resolve_governance.py",
    ]
    
    for req_file in required_files:
        fpath = gov_dir / req_file
        assert fpath.exists(), f"Missing required governance file: {req_file}"
        assert fpath.stat().st_size > 100, f"Governance file is empty or too short: {req_file}"


def test_ci_gate_scanner_detects_violations():
    """Test that ci_gate scanner correctly identifies rule violations."""
    import importlib.util
    root_dir = Path(__file__).resolve().parent.parent
    ci_gate_path = root_dir / ".governance" / "scripts" / "ci_gate.py"
    
    spec = importlib.util.spec_from_file_location("ci_gate", ci_gate_path)
    ci_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_gate)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test Stub Violation
        stub_file = Path(tmpdir) / "test_stub.py"
        # Avoid putting the test word in the file name so the non-test filter activates
        bad_strategy = Path(tmpdir) / "oms" / "sample_rule.py"
        bad_strategy.parent.mkdir(parents=True, exist_ok=True)
        
        bad_strategy.write_text("""
def calculate_margin(equity):
    # TODO: implement margin logic
    realized_pnl = float(equity)
    return realized_pnl
""", encoding="utf-8")

        violations = ci_gate.scan_file_for_governance_violations(bad_strategy)
        assert len(violations) >= 2
        rule_ids = [v.rule_id for v in violations]
        assert "STUB_TODO" in rule_ids
        assert "RULE_05_FLOAT_ACCOUNTING" in rule_ids
