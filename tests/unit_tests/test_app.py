"""Unit tests for app module-level initialization."""

import os

import pytest

# Import app module to trigger TF_CPP_MIN_LOG_LEVEL setup
from src import app  # noqa: F401


class TestModuleLevel:

    def test_tf_cpp_log_level_set_at_import(self):
        """TF_CPP_MIN_LOG_LEVEL is set when app module is imported."""
        # The import at the top of this file already triggers the setdefault.
        # Verify it's set (setdefault won't overwrite if already present).
        assert "TF_CPP_MIN_LOG_LEVEL" in os.environ
        # Value should be '1' (the default) or whatever was already set
        assert os.environ["TF_CPP_MIN_LOG_LEVEL"] in ("1", "2", "3")
