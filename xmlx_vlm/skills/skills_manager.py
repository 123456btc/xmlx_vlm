# SPDX-License-Identifier: Apache-2.0
"""
Skills Manager -- Parses, registers, and executes agent skills.

Manages skill folders containing SKILL.md and companion scripts.
Ensures skills comply with security and formatting standards.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmlx_vlm.skills.skills_guard import SkillsGuard

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = os.path.expanduser("~/.cache/xmlx_vlm/skills")


@dataclass
class SkillMetadata:
    """Parsed frontmatter and attributes of a skill."""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "agent"
    created_by: str = "agent"  # "agent" | "user" | "builtin"
    platforms: List[str] = field(default_factory=lambda: ["macos", "linux", "windows"])
    tools_required: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    directory_path: Optional[str] = None
    body_content: str = ""


class SkillsManager:
    """
    Discovers, verifies, and registers skill bundles.
    """

    def __init__(self, skills_dir: Optional[str] = None, guard: Optional[SkillsGuard] = None):
        self.skills_dir = Path(skills_dir or DEFAULT_SKILLS_DIR)
        self.guard = guard or SkillsGuard()
        self.skills: Dict[str, SkillMetadata] = {}
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.discover()

    @staticmethod
    def parse_skill_md(content: str, directory_path: Optional[str] = None) -> SkillMetadata:
        """Parse frontmatter YAML and markdown body from SKILL.md."""
        frontmatter: Dict[str, Any] = {}
        body = content

        # Look for YAML frontmatter between --- and ---
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if fm_match:
            fm_text, body = fm_match.group(1), fm_match.group(2)
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if v.startswith("[") and v.endswith("]"):
                        # Basic list parse
                        items = [x.strip().strip("\"'") for x in v[1:-1].split(",") if x.strip()]
                        frontmatter[k] = items
                    else:
                        frontmatter[k] = v

        name = frontmatter.get("name", "unnamed_skill")
        desc = frontmatter.get("description", "")
        version = frontmatter.get("version", "1.0.0")
        author = frontmatter.get("author", "agent")
        created_by = frontmatter.get("created_by", "agent")
        platforms = frontmatter.get("platforms", ["macos", "linux", "windows"])
        if isinstance(platforms, str):
            platforms = [platforms]
        tools_req = frontmatter.get("tools_required", [])
        if isinstance(tools_req, str):
            tools_req = [tools_req]
        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        return SkillMetadata(
            name=name,
            description=desc,
            version=version,
            author=author,
            created_by=created_by,
            platforms=platforms,
            tools_required=tools_req,
            tags=tags,
            directory_path=directory_path,
            body_content=body.strip(),
        )

    def discover(self) -> Dict[str, SkillMetadata]:
        """Scan skills directory for valid skill subdirectories."""
        self.skills.clear()
        if not self.skills_dir.exists():
            return self.skills

        for entry in self.skills_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                skill_md_path = entry / "SKILL.md"
                if skill_md_path.exists():
                    try:
                        content = skill_md_path.read_text(encoding="utf-8")
                        meta = self.parse_skill_md(content, directory_path=str(entry))
                        # Run security scan
                        scan_res = self.guard.scan_directory(entry, skill_name=meta.name)
                        if scan_res.is_allowed:
                            self.skills[meta.name] = meta
                        else:
                            logger.warning(
                                "Skill [%s] blocked due to security scan: %s",
                                meta.name,
                                scan_res.verdict,
                            )
                    except Exception as e:
                        logger.error("Failed to load skill at %s: %s", entry, e)

        return self.skills

    def register_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        created_by: str = "agent",
        companion_scripts: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """
        Create and persist a new skill bundle. Returns (success, message).
        """
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        fm_lines = [
            "---",
            f"name: {name}",
            f"description: \"{description[:120]}\"",
            "version: 1.0.0",
            f"created_by: {created_by}",
            "---",
            "",
            instructions,
        ]
        skill_md_content = "\n".join(fm_lines)

        # Write scripts if any
        if companion_scripts:
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            for sname, scontent in companion_scripts.items():
                spath = scripts_dir / sname
                spath.write_text(scontent, encoding="utf-8")

        # Write SKILL.md
        skill_md_path = skill_dir / "SKILL.md"
        skill_md_path.write_text(skill_md_content, encoding="utf-8")

        # Security check before activating
        scan = self.guard.scan_directory(skill_dir, skill_name=name)
        if not scan.is_allowed:
            # Clean up dangerous directory
            try:
                for root, _, files in os.walk(skill_dir, topdown=False):
                    for f in files:
                        (Path(root) / f).unlink(missing_ok=True)
                    Path(root).rmdir()
            except Exception:
                pass
            return False, f"Skill creation rejected by security guard: {scan.verdict}"

        meta = self.parse_skill_md(skill_md_content, directory_path=str(skill_dir))
        self.skills[name] = meta
        return True, f"Skill '{name}' successfully created and verified."

    def build_system_prompt_skills_summary(self) -> str:
        """Generate a concise summary of available skills for injection into the system prompt."""
        if not self.skills:
            return ""

        lines = ["## Available Skills:"]
        for name, meta in sorted(self.skills.items()):
            lines.append(f"- **{name}**: {meta.description}")
        return "\n".join(lines)
