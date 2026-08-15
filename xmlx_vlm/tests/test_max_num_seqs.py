"""Tests for max_num_seqs server concurrency limiter."""

import os
import pytest
from unittest.mock import patch

from xmlx_vlm.config import DEFAULT_MAX_NUM_SEQS, get_max_num_seqs


def test_get_max_num_seqs_default():
    with patch.dict(os.environ, {}, clear=True):
        assert get_max_num_seqs() == DEFAULT_MAX_NUM_SEQS


def test_get_max_num_seqs_env_override():
    with patch.dict(os.environ, {"XMLX_VLM_MAX_NUM_SEQS": "8"}):
        assert get_max_num_seqs() == 8

    # Ensure min bounded at 1
    with patch.dict(os.environ, {"XMLX_VLM_MAX_NUM_SEQS": "0"}):
        assert get_max_num_seqs() == 1

    # Invalid input falls back to default
    with patch.dict(os.environ, {"XMLX_VLM_MAX_NUM_SEQS": "invalid"}):
        assert get_max_num_seqs() == DEFAULT_MAX_NUM_SEQS
