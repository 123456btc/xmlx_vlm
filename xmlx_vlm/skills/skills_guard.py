# SPDX-License-Identifier: Apache-2.0
"""
Skills Guard -- Security scanner for skills scripts and instructions.

Inspects skills for known threat patterns:
1. Exfiltration (curl/wget with secret environment variables)
2. Destructive filesystem operations (e.g. rm -rf /, dd overwrite)
3. Prompt injection and jailbreak payloads
4. Unauthorized persistence mechanisms

Applies trust-aware install and execution policies.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ScanFinding:
    """Security finding detected during scan."""

    pattern_id: str
    severity: str  # "critical" | "high" | "medium" | "low"
    category: str  # "exfiltration" | "destructive" | "injection" | "persistence"
    matched_text: str
    description: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None


@dataclass
class ScanResult:
    """Consolidated security verdict for a skill."""

    skill_name: str
    source_trust: str  # "builtin" | "trusted" | "community" | "agent_created"
    verdict: str  # "safe" | "caution" | "dangerous"
    findings: List[ScanFinding] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        """Evaluate whether the skill is permitted to load under standard policy."""
        if self.source_trust == "builtin":
            return True
        if self.source_trust == "trusted":
            return self.verdict in ("safe", "caution")
        if self.source_trust == "agent_created":
            return self.verdict in ("safe", "caution")
        # Community requires safe
        return self.verdict == "safe"


# Threat patterns: (regex, pattern_id, severity, category, description)
THREAT_PATTERNS = [
    # Exfiltration: Shell commands sending secrets out
    (
        r"(?:curl|wget|fetch|http)\s+[^\n]*\$\{?(?:[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)[A-Z0-9_]*)\}?",
        "env_exfiltration",
        "critical",
        "exfiltration",
        "Command interpolates sensitive environment variable into network request",
    ),
    # Destructive: dangerous rm / format
    (
        r"rm\s+-(?:r|f|rf|fr)\s+/(?:\s|$|\*)",
        "destructive_root_rm",
        "critical",
        "destructive",
        "Attempting recursive removal of filesystem root",
    ),
    (
        r"(?:mkfs|fdisk|dd\s+if=/dev/zero\s+of=/dev/)",
        "destructive_disk_format",
        "critical",
        "destructive",
        "Dangerous raw disk write or filesystem formatting command",
    ),
    # Persistence: cron / systemd unauthorized modifications
    (
        r"(?:crontab\s+-r|echo\s+[^\n]*>>\s*/etc/crontab)",
        "persistence_cron_tamper",
        "high",
        "persistence",
        "Modifying system crontab for persistence",
    ),
    # Injection: Overriding safety directives
    (
        r"(?:ignore\s+all\s+(?:previous|above)\s+instructions|system\s+prompt\s+override)",
        "prompt_injection_jailbreak",
        "high",
        "injection",
        "Potential prompt injection attempting to override system constraints",
    ),
]


class SkillsGuard:
    """
    Scans skill markdown and script files against threat patterns.
    """

    def __init__(self, custom_patterns: Optional[List[Tuple[str, str, str, str, str]]] = None):
        raw_patterns = THREAT_PATTERNS + (custom_patterns or [])
        self.compiled_patterns = [
            (re.compile(regex, re.IGNORECASE), pid, sev, cat, desc)
            for regex, pid, sev, cat, desc in raw_patterns
        ]

    def scan_content(
        self,
        content: str,
        skill_name: str = "unnamed_skill",
        source_trust: str = "community",
        file_path: Optional[str] = None,
    ) -> ScanResult:
        """Scan a raw string for security findings."""
        findings: List[ScanFinding] = []
        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            for regex, pid, sev, cat, desc in self.compiled_patterns:
                match = regex.search(line)
                if match:
                    findings.append(
                        ScanFinding(
                            pattern_id=pid,
                            severity=sev,
                            category=cat,
                            matched_text=match.group(0),
                            description=desc,
                            line_number=idx,
                            file_path=file_path,
                        )
                    )

        # Derive verdict
        has_critical = any(f.severity == "critical" for f in findings)
        has_high = any(f.severity == "high" for f in findings)
        if has_critical or has_high:
            verdict = "dangerous"
        elif findings:
            verdict = "caution"
        else:
            verdict = "safe"

        return ScanResult(
            skill_name=skill_name,
            source_trust=source_trust,
            verdict=verdict,
            findings=findings,
        )

    def scan_directory(
        self,
        skill_dir: Path | str,
        skill_name: Optional[str] = None,
        source_trust: str = "community",
    ) -> ScanResult:
        """Scan all markdown and script files in a skill directory."""
        path = Path(skill_dir)
        name = skill_name or path.name
        all_findings: List[ScanFinding] = []

        if not path.exists():
            return ScanResult(skill_name=name, source_trust=source_trust, verdict="safe")

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith((".md", ".py", ".sh", ".js", ".json", ".yaml", ".yml")):
                    fpath = Path(root) / file
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="ignore")
                        res = self.scan_content(
                            content=text,
                            skill_name=name,
                            source_trust=source_trust,
                            file_path=str(fpath.relative_to(path)),
                        )
                        all_findings.extend(res.findings)
                    except Exception:
                        continue

        has_critical = any(f.severity == "critical" for f in all_findings)
        has_high = any(f.severity == "high" for f in all_findings)
        if has_critical or has_high:
            verdict = "dangerous"
        elif all_findings:
            verdict = "caution"
        else:
            verdict = "safe"

        return ScanResult(
            skill_name=name,
            source_trust=source_trust,
            verdict=verdict,
            findings=all_findings,
        )
