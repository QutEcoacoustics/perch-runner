"""End-to-end tests for CLI exit codes."""

import subprocess
import sys
from pathlib import Path


class TestCLIExitCodes:
    """E2E: Test that the app exits with appropriate codes for various error conditions."""

    REPO_ROOT = Path(__file__).resolve().parents[2]

    def _run_app(self, *args):
        """Run app.py as a subprocess and return the result."""
        cmd = [sys.executable, '-m', 'src.app'] + list(args)
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(self.REPO_ROOT),
        )

    def test_missing_source_exits_nonzero(self):
        result = self._run_app(
            'analyze',
            '--embed', '--source', '/nonexistent/path', '--output', '/tmp'
        )
        assert result.returncode != 0

    def test_no_action_exits_nonzero(self):
        # No --embed or --classify should fail config validation.
        result = self._run_app('analyze', '--source', '/tmp', '--output', '/tmp')
        assert result.returncode != 0

    def test_invalid_model_exits_nonzero(self):
        result = self._run_app(
            'analyze',
            '--embed', '--model_choice', 'bogus_model'
        )
        assert result.returncode != 0
