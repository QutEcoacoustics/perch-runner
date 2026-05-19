"""End-to-end tests for app subcommands (version, config)."""

import importlib
import json
import os
from unittest.mock import patch

import pytest

from src.app import main
from src.config import default_config


class TestVersionCommand:
    """E2E: Test version subcommand output."""

    def test_version_prints_and_exits(self, capsys):
        with patch.dict(os.environ, {"APP_VERSION": "dev"}):
            # src.version reads APP_VERSION at import time, and src.app imports
            # that module-level value. Reload both inside this context so the
            # command reflects the patched environment.
            import src.version
            importlib.reload(src.version)
            import src.app
            importlib.reload(src.app)
            from src.app import main
            with patch("sys.argv", ["app", "version"]):
                main()
        output = capsys.readouterr().out
        assert "perch-runner dev" in output
        assert "perch-hoplite" in output
        assert "perch_8" in output
        assert "perch_v2" in output


class TestConfigCommand:
    """E2E: Test config subcommand output."""

    def test_config_prints_default_config_and_exits(self, capsys):
        with patch("sys.argv", ["app", "config"]):
            main()

        output = capsys.readouterr().out
        printed = json.loads(output)
        assert printed == default_config
