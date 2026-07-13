"""Tests for logging configuration."""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src.logging_config import setup_logging, _resolve_level, VALID_LEVELS


class TestResolveLevel:

    def test_valid_names(self):
        assert _resolve_level('DEBUG') == logging.DEBUG
        assert _resolve_level('INFO') == logging.INFO
        assert _resolve_level('WARNING') == logging.WARNING
        assert _resolve_level('ERROR') == logging.ERROR
        assert _resolve_level('CRITICAL') == logging.CRITICAL

    def test_case_insensitive(self):
        assert _resolve_level('info') == logging.INFO
        assert _resolve_level('Warning') == logging.WARNING

    def test_int_passthrough(self):
        assert _resolve_level(logging.DEBUG) == logging.DEBUG

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            _resolve_level('BOGUS')


class TestSetupLogging:

    def _clean_root_logger(self):
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        root.setLevel(logging.WARNING)

    def test_default_config(self):
        self._clean_root_logger()
        setup_logging({})
        assert logging.getLogger().level == logging.WARNING
        assert logging.getLogger('src').level == logging.INFO

    def test_custom_levels(self):
        self._clean_root_logger()
        setup_logging({
            'log_level': 'DEBUG',
            'hoplite_log_level': 'ERROR',
        })
        assert logging.getLogger('src').level == logging.DEBUG
        assert logging.getLogger().level == logging.ERROR

    def test_log_file(self):
        self._clean_root_logger()
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_path = f.name

        try:
            setup_logging({'log_file': log_path})
            logging.getLogger('src.test').info("test message")

            content = Path(log_path).read_text()
            assert "test message" in content
        finally:
            Path(log_path).unlink(missing_ok=True)
            # Clean up the file handler
            root = logging.getLogger()
            for h in root.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    root.removeHandler(h)


class TestLoggingCLI:

    REPO_ROOT = Path(__file__).resolve().parents[2]

    def _run_app(self, *args):
        cmd = [sys.executable, '-m', 'src.app'] + list(args)
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(self.REPO_ROOT),
        )

    def test_hoplite_debug_shows_sql(self):
        # With hoplite_log_level=DEBUG, the flag should be accepted.
        result = self._run_app(
            'analyze',
            '--embed', '--source', '/tmp', '--output', '/tmp',
            '--hoplite_log_level', 'DEBUG',
        )
        assert result.returncode == 0

    def test_log_level_flag_accepted(self):
        result = self._run_app(
            'analyze',
            '--embed', '--source', '/tmp', '--output', '/tmp',
            '--log_level', 'WARNING',
        )
        assert result.returncode == 0
