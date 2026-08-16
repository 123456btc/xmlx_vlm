#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CI Quality & Governance Gate — Automated static scan & verification runner.

Enforces Constitutional Governance Rules:
1. Rule 05: No raw float usage in financial/OMS accounting paths.
2. Rule 06 & 18: No fake placeholder implementations or stubs.
3. Rule 09: No raw datetime.now() in strategy or OMS execution logic.
4. Rule 12 & 15: Verifiable test runner integration.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional


class Violation(NamedTuple):
    file_path: str
    line_number: int
    rule_id: str
    rule_title: str
    message: str
    code_snippet: str


def scan_file_for_governance_violations(file_path: Path) -> List[Violation]:
    """Scan a single python file for constitution violations."""
    violations: List[Violation] = []
    
    # Skip test files and migration tools for some strict checks
    is_test_file = "test" in file_path.name or "tests" in file_path.parts
    is_financial_path = any(part in file_path.parts for part in ("oms", "store", "risk", "accounting"))
    is_strategy_path = any(part in file_path.parts for part in ("strategies", "decision", "agent"))

    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
    except Exception as e:
        return [Violation(str(file_path), 0, "READ_ERR", "File Read Error", str(e), "")]

    # Regex patterns for stubs & fake implementations
    stub_patterns = [
        (re.compile(r"TODO:\s*implement", re.IGNORECASE), "STUB_TODO", "Rule 18: Unfinished TODO implementation stub"),
        (re.compile(r"placeholder\s*=\s*(0\.0|0|None)", re.IGNORECASE), "STUB_VAL", "Rule 14: Placeholder financial value"),
    ]

    for line_idx, line in enumerate(lines, start=1):
        # 1. Stub checks
        if not is_test_file:
            for pattern, rule_id, rule_title in stub_patterns:
                if pattern.search(line):
                    violations.append(
                        Violation(
                            str(file_path),
                            line_idx,
                            rule_id,
                            rule_title,
                            f"Found forbidden stub pattern: {line.strip()}",
                            line.strip(),
                        )
                    )

        # 2. Raw wall-clock time in strategy/OMS execution
        if is_strategy_path or is_financial_path:
            if not is_test_file:
                if "datetime.now()" in line and "ClockProvider" not in content and "# allow-wall-clock" not in line:
                    violations.append(
                        Violation(
                            str(file_path),
                            line_idx,
                            "RULE_09_WALLCLOCK",
                            "Rule 09: Wall-Clock Determinism Violation",
                            "Direct call to datetime.now() in strategy/financial path. Use ClockProvider.",
                            line.strip(),
                        )
                    )

    # AST checks for financial paths
    if is_financial_path and not is_test_file:
        try:
            tree = ast.parse(content, filename=str(file_path))
            for node in ast.walk(tree):
                # Check for float(0.0) or float conversions on balance/price fields
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "float":
                        # Check if arg is an integer/string or part of accounting
                        line_no = getattr(node, "lineno", 0)
                        line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                        if any(k in line_text.lower() for k in ("equity", "balance", "margin", "pnl", "cash", "notional")):
                            if "# allow-float" not in line_text:
                                violations.append(
                                    Violation(
                                        str(file_path),
                                        line_no,
                                        "RULE_05_FLOAT_ACCOUNTING",
                                        "Rule 05: Raw Float Accounting Violation",
                                        "Detected float() conversion in financial/balance accounting path. Use Decimal.",
                                        line_text.strip(),
                                    )
                                )
        except Exception:
            pass

    return violations


def run_tests(scope: Optional[str] = None) -> bool:
    """Execute pytest suite and return success status."""
    cmd = [sys.executable, "-m", "pytest"]
    if scope:
        cmd.append(scope)
    else:
        cmd.extend(["tests"])
    
    print(f"\n🚀 Running Verifiable Test Suite: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="x9/xmlx_vlm CI Governance Gate")
    parser.add_argument("--quick", action="store_true", help="Run static scan on changed files only")
    parser.add_argument("--scope", type=str, help="Scope scan and tests to a specific directory or file")
    parser.add_argument("--full", action="store_true", help="Full governance scan across all files and tests")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pytest (scan only)")
    
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent.parent
    target_dir = root_dir / (args.scope or "xmlx_vlm")

    print(f"🛡️  [CI GOVERNANCE GATE] Starting scan in: {target_dir}")
    all_violations: List[Violation] = []

    if target_dir.is_file():
        files_to_scan = [target_dir]
    elif target_dir.is_dir():
        files_to_scan = list(target_dir.rglob("*.py"))
    else:
        print(f"❌ Target path does not exist: {target_dir}")
        return 1

    for py_file in files_to_scan:
        if "__pycache__" in py_file.parts or ".venv" in py_file.parts or ".git" in py_file.parts:
            continue
        v = scan_file_for_governance_violations(py_file)
        all_violations.extend(v)

    # Print results
    if all_violations:
        print(f"\n❌ [GATE BLOCKED] Found {len(all_violations)} Constitution Violations:\n")
        for v in all_violations:
            rel_path = os.path.relpath(v.file_path, root_dir)
            print(f"  🔴 [{v.rule_id}] {v.rule_title}")
            print(f"     File: {rel_path}:{v.line_number}")
            print(f"     Detail: {v.message}")
            if v.code_snippet:
                print(f"     Code:   `{v.code_snippet}`")
            print()
        print("💡 Resolve all violations before merging or claiming task completion.")
        return 2

    print(f"✅ [STATIC CHECK PASSED] Scanned {len(files_to_scan)} files with 0 governance violations.")

    # Run tests unless explicitly skipped or quick
    if not args.skip_tests and (args.full or args.scope or not args.quick):
        test_scope = None
        if args.scope and "tests" in args.scope:
            test_scope = args.scope
        success = run_tests(test_scope)
        if not success:
            print("\n❌ [GATE BLOCKED] Pytest execution failed.")
            return 1
        print("\n✅ [ALL TESTS PASSED] Verifiable test evidence confirmed.")

    print("\n🎉 [CI GATE SUCCESS] All governance requirements satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
