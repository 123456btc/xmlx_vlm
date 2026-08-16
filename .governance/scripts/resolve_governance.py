#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Governance Resolver — Resolves matching governance policies for any given file or directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MAPPING = [
    ("agent_core", "ENGINEERING_CONSTITUTION.md", "Tool mutation isolation, loop detection, anti-overtrading."),
    ("oms", "ENGINEERING_CONSTITUTION.md §1, §2", "OMS SSOT gateway, daily loss limit, Decimal accounting, correlation IDs."),
    ("decision", "AI_GOVERNANCE_OPERATING_MODEL.md", "Multi-Agent consensus debate, Risk Officer veto, ATR stop validation."),
    ("agent", "ENGINEERING_CONSTITUTION.md §1", "Strategy Analyst vs Risk Officer separation, reflection injection."),
    ("market_service", "ENGINEERING_CONSTITUTION.md §3", "Memory state machine, bounded queues, websocket reconnection backoff."),
    ("store", "ENGINEERING_CONSTITUTION.md §5", "SQLite session persistence, crash recovery, reflection audit logs."),
    ("skills", "ENGINEERING_CONSTITUTION.md §4", "Skills security scanner, exfiltration & injection threat defense."),
    ("auth", "ENGINEERING_CONSTITUTION.md §5", "KMS vault isolation, zero plaintext secret logs, Paper mode default."),
]


def resolve(target_path: str):
    root_dir = Path(__file__).resolve().parent.parent.parent
    gov_dir = root_dir / ".governance"
    
    print(f"\n🔍 Resolving Governance Policies for: {target_path}\n" + "-" * 60)
    matched = False
    for keyword, policy, focus in MAPPING:
        if keyword in target_path:
            matched = True
            print(f"📌 Domain / Component: {keyword}")
            print(f"📜 Mandatory Policies:  {policy}")
            print(f"🛡️  Core Invariants:     {focus}")
            print(f"📁 Reference Doc:       {gov_dir}/{policy.split()[0]}")
            print("-" * 60)

    if not matched:
        print("📌 Global Default Policies:")
        print("📜 Mandatory Policies:  ENGINEERING_CONSTITUTION.md & GOVERNANCE_PRIMACY.md")
        print("🛡️  Core Invariants:     32 Non-negotiables, Decimal accounting, TDD evidence.")
        print(f"📁 Reference Doc:       {gov_dir}/ENGINEERING_CONSTITUTION.md")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Resolve governance policy for target path")
    parser.add_argument("--scope", type=str, required=True, help="Target file or directory path")
    args = parser.parse_args()
    resolve(args.scope)


if __name__ == "__main__":
    main()
