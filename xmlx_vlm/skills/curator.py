# SPDX-License-Identifier: Apache-2.0
"""
Skills Curator -- Automatic skill lifecycle and maintenance orchestrator.

Tracks skill usage telemetry, pins important skills, and auto-archives stale
agent-created skills to prevent system prompt and skill catalog bloat.
Ensures zero data loss by archiving to `.archive/` instead of hard deletion.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from xmlx_vlm.skills.skills_manager import SkillsManager

logger = logging.getLogger(__name__)


@dataclass
class SkillUsage:
    """Usage statistics and lifecycle state for a single skill."""

    name: str
    use_count: int = 0
    created_at: float = 0.0
    last_used_at: float = 0.0
    pinned: bool = False
    state: str = "active"  # "active" | "stale" | "archived"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillUsage":
        return cls(
            name=data.get("name", ""),
            use_count=data.get("use_count", 0),
            created_at=data.get("created_at", time.time()),
            last_used_at=data.get("last_used_at", data.get("created_at", time.time())),
            pinned=data.get("pinned", False),
            state=data.get("state", "active"),
        )


class SkillsCurator:
    """
    Monitors skill usage and performs background maintenance.
    """

    def __init__(
        self,
        manager: SkillsManager,
        stale_after_days: float = 30.0,
        archive_after_days: float = 90.0,
    ):
        self.manager = manager
        self.stale_after_days = stale_after_days
        self.archive_after_days = archive_after_days
        self.usage_file = self.manager.skills_dir / ".usage.json"
        self.archive_dir = self.manager.skills_dir / ".archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry: Dict[str, SkillUsage] = {}
        self._load_telemetry()

    def _load_telemetry(self) -> None:
        """Load persistent telemetry from .usage.json."""
        if self.usage_file.exists():
            try:
                data = json.loads(self.usage_file.read_text(encoding="utf-8"))
                for k, v in data.items():
                    self.telemetry[k] = SkillUsage.from_dict(v)
            except Exception as e:
                logger.warning("Failed to load skills telemetry: %s", e)

        # Sync with discovered skills
        now = time.time()
        for name in self.manager.skills:
            if name not in self.telemetry:
                self.telemetry[name] = SkillUsage(
                    name=name,
                    created_at=now,
                    last_used_at=now,
                    pinned=False,
                    state="active",
                )

    def _save_telemetry(self) -> None:
        """Persist telemetry state."""
        try:
            payload = {k: v.to_dict() for k, v in self.telemetry.items()}
            self.usage_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save skills telemetry: %s", e)

    def record_usage(self, skill_name: str) -> None:
        """Record that a skill was invoked by the agent."""
        if skill_name in self.telemetry:
            self.telemetry[skill_name].use_count += 1
            self.telemetry[skill_name].last_used_at = time.time()
            self.telemetry[skill_name].state = "active"
            self._save_telemetry()

    def pin_skill(self, skill_name: str) -> bool:
        """Pin a skill to exempt it from auto-archiving."""
        if skill_name in self.telemetry:
            self.telemetry[skill_name].pinned = True
            self._save_telemetry()
            return True
        return False

    def unpin_skill(self, skill_name: str) -> bool:
        """Unpin a skill."""
        if skill_name in self.telemetry:
            self.telemetry[skill_name].pinned = False
            self._save_telemetry()
            return True
        return False

    def evaluate_and_archive(self) -> List[str]:
        """
        Scan skills and archive stale, unpinned agent-created skills.
        Returns a list of archived skill names.
        """
        now = time.time()
        archived_skills: List[str] = []

        for name, meta in list(self.manager.skills.items()):
            usage = self.telemetry.get(name)
            if not usage:
                continue

            # Only agent-created skills are candidates for auto-archival
            if meta.created_by != "agent":
                continue

            # Pinned skills are never auto-archived
            if usage.pinned:
                continue

            days_idle = (now - usage.last_used_at) / (24 * 3600)

            if days_idle >= self.archive_after_days:
                # Move to archive directory
                src_dir = self.manager.skills_dir / name
                dst_dir = self.archive_dir / name
                if src_dir.exists():
                    try:
                        if dst_dir.exists():
                            shutil.rmtree(dst_dir)
                        shutil.move(str(src_dir), str(dst_dir))
                        usage.state = "archived"
                        archived_skills.append(name)
                        logger.info("Archived stale skill [%s] (idle for %.1f days)", name, days_idle)
                    except Exception as e:
                        logger.error("Failed to archive skill %s: %s", name, e)

            elif days_idle >= self.stale_after_days:
                usage.state = "stale"

        self._save_telemetry()
        # Refresh manager catalog
        self.manager.discover()
        return archived_skills

    def restore_skill(self, skill_name: str) -> Tuple[bool, str]:
        """Restore an archived skill back to the active catalog."""
        archived_path = self.archive_dir / skill_name
        if not archived_path.exists():
            return False, f"Archived skill '{skill_name}' not found."

        active_path = self.manager.skills_dir / skill_name
        try:
            if active_path.exists():
                shutil.rmtree(active_path)
            shutil.move(str(archived_path), str(active_path))
            if skill_name in self.telemetry:
                self.telemetry[skill_name].state = "active"
                self.telemetry[skill_name].last_used_at = time.time()
                self._save_telemetry()
            self.manager.discover()
            return True, f"Skill '{skill_name}' successfully restored."
        except Exception as e:
            return False, f"Failed to restore skill: {e}"
