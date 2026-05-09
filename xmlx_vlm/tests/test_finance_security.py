# SPDX-License-Identifier: Apache-2.0
"""Tests for finance-mode security hardening."""

import os
import unittest
from unittest.mock import patch

from xmlx_vlm.mcp import security
from xmlx_vlm.mcp.executor import ToolExecutor
from xmlx_vlm.mcp.tools import shell, _scrub_pii


class TestFinanceMode(unittest.TestCase):
    def test_finance_mode_blocks_shell(self):
        """Shell tool is disabled when finance mode is on."""
        with patch.dict(os.environ, {"XMLX_VLM_FINANCE_MODE": "1"}, clear=False):
            # Re-evaluate finance mode flag in tools module
            import importlib
            from xmlx_vlm.mcp import tools

            importlib.reload(tools)
            result = tools.shell("ls -la")
            self.assertIn("disabled", result.lower())

    def test_finance_mode_default_tool_whitelist(self):
        """Finance mode defaults to safe read-only tools."""
        with patch.dict(os.environ, {"XMLX_VLM_FINANCE_MODE": "1"}, clear=False):
            import importlib

            importlib.reload(security)
            allowed = security.get_allowed_tools()
            self.assertIsNotNone(allowed)
            self.assertIn("read_file", allowed)
            self.assertIn("list_dir", allowed)
            self.assertIn("search_files", allowed)
            self.assertNotIn("shell", allowed)
            self.assertNotIn("write_file", allowed)

    def test_tool_whitelist_blocks_disallowed(self):
        """Executor rejects tools not in whitelist."""
        with patch.dict(
            os.environ,
            {"XMLX_VLM_ALLOWED_TOOLS": "read_file,list_dir"},
            clear=False,
        ):
            import importlib

            importlib.reload(security)
            executor = ToolExecutor()
            result = executor.call("shell", {"command": "ls"})
            self.assertIn("not in the allowed tools whitelist", result)

    def test_shell_hardening_no_metacharacters(self):
        """Shell rejects commands with metacharacters even outside finance mode."""
        import importlib
        from xmlx_vlm.mcp import tools

        importlib.reload(tools)
        result = tools.shell("git status; rm -rf /")
        self.assertIn("metacharacters", result)

    def test_shell_hardening_shell_false(self):
        """Shell uses shell=False (argument list, not string)."""
        import importlib
        from xmlx_vlm.mcp import tools

        importlib.reload(tools)
        result = tools.shell("pwd")
        # Should succeed (or at least not complain about syntax)
        self.assertNotIn("metacharacters", result)

    def test_pii_scrubbing_credit_card(self):
        """Finance mode scrubs credit card numbers from output."""
        with patch.dict(os.environ, {"XMLX_VLM_FINANCE_MODE": "1"}, clear=False):
            import importlib
            from xmlx_vlm.mcp import tools

            importlib.reload(tools)
            text = "Card: 4111111111111111 and 5500000000000004"
            scrubbed = tools._scrub_pii(text)
            self.assertNotIn("4111111111111111", scrubbed)
            self.assertIn("[CREDIT_CARD]", scrubbed)

    def test_pii_scrubbing_phone(self):
        """Finance mode scrubs phone numbers."""
        with patch.dict(os.environ, {"XMLX_VLM_FINANCE_MODE": "1"}, clear=False):
            import importlib
            from xmlx_vlm.mcp import tools

            importlib.reload(tools)
            text = "Contact: 13800138000"
            scrubbed = tools._scrub_pii(text)
            self.assertNotIn("13800138000", scrubbed)
            self.assertIn("[PHONE]", scrubbed)

    def test_read_only_blocks_write(self):
        """Read-only mode blocks write_file."""
        with patch.dict(os.environ, {"MLX_MCP_READONLY": "true"}, clear=False):
            import importlib

            importlib.reload(security)
            executor = ToolExecutor()
            result = executor.call("write_file", {"path": "/tmp/x.txt", "content": "x"})
            self.assertIn("read-only", result.lower())

    def test_config_integrity_mismatch(self):
        """Config integrity check fails on hash mismatch."""
        with patch.dict(
            os.environ,
            {"XMLX_VLM_MCP_CONFIG_HASH": "deadbeef" * 8},
            clear=False,
        ):
            import importlib

            importlib.reload(security)
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                f.write('{"servers":{}}')
                path = f.name
            self.assertFalse(security.verify_config_integrity(path))
            os.unlink(path)

    def test_config_integrity_match(self):
        """Config integrity check passes on correct hash."""
        import hashlib
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            content = '{"servers":{}}'
            f.write(content)
            path = f.name
            expected = hashlib.sha256(content.encode()).hexdigest()

        with patch.dict(
            os.environ,
            {"XMLX_VLM_MCP_CONFIG_HASH": expected},
            clear=False,
        ):
            import importlib

            importlib.reload(security)
            self.assertTrue(security.verify_config_integrity(path))
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
