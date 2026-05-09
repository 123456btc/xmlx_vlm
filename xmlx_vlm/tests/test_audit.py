# SPDX-License-Identifier: Apache-2.0
"""Tests for the audit logging module."""

import json
import os
import tempfile
import unittest

from xmlx_vlm.audit import AuditLogger, _sha256_hex


class TestAuditLogger(unittest.TestCase):
    def setUp(self):
        self.fd, self.path = tempfile.mkstemp(suffix=".log")
        self.logger = AuditLogger(path=self.path, enabled=True)

    def tearDown(self):
        os.close(self.fd)
        os.unlink(self.path)

    def test_sha256_hex(self):
        h = _sha256_hex("hello")
        self.assertEqual(len(h), 16)
        self.assertNotEqual(_sha256_hex("hello"), _sha256_hex("world"))

    def test_log_tool_call(self):
        self.logger.log_tool_call(
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            result="hello world",
            source_ip="127.0.0.1",
            session_id="sess1",
        )
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["tool"], "read_file")
        self.assertEqual(record["src_ip"], "127.0.0.1")
        self.assertEqual(record["session"], "sess1")
        self.assertIn("args_hash", record)
        self.assertIn("result_hash", record)

    def test_log_security_event(self):
        self.logger.log_security_event("blocked_tool", "tool=shell")
        with open(self.path, "r") as f:
            record = json.loads(f.readline())
        self.assertEqual(record["event"], "security")
        self.assertEqual(record["type"], "blocked_tool")
        self.assertEqual(record["detail"], "tool=shell")

    def test_multiple_records(self):
        self.logger.log_tool_call("t1", {}, "r1")
        self.logger.log_tool_call("t2", {}, "r2")
        with open(self.path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
