"""Tests for auto-worker calculation and exit code handling."""

import subprocess
import sys

import pytest

from src.resources import RAM_BASE_GB, RAM_PER_WORKER_GB, compute_workers


class TestComputeWorkers:
    """Test the compute_workers function."""

    def test_explicit_int(self):
        assert compute_workers(4) == 4

    def test_explicit_string_int(self):
        assert compute_workers('2') == 2

    def test_explicit_minimum_is_1(self):
        assert compute_workers(0) == 1
        assert compute_workers(-1) == 1

    def test_explicit_maximum_is_uncapped(self):
        # Explicit values are not capped (user knows best)
        assert compute_workers(16) == 16

    def test_auto_plenty_of_ram(self):
        # 20GB available → (20 - 2) / 1.5 = 12 → capped at 8
        result = compute_workers('auto', available_ram_gb=20.0)
        assert result == 8

    def test_auto_moderate_ram(self):
        # 8GB available → (8 - 2) / 1.5 = 4
        result = compute_workers('auto', available_ram_gb=8.0)
        assert result == 4

    def test_auto_low_ram(self):
        # 4GB available → (4 - 2) / 1.5 = 1.3 → 1
        result = compute_workers('auto', available_ram_gb=4.0)
        assert result == 1

    def test_auto_very_low_ram(self):
        # 2GB available → (2 - 2) / 1.5 = 0 → minimum 1
        result = compute_workers('auto', available_ram_gb=2.0)
        assert result == 1

    def test_auto_negative_usable(self):
        # Less than base → still returns 1
        result = compute_workers('auto', available_ram_gb=1.0)
        assert result == 1

    def test_auto_exact_boundary(self):
        # Exactly enough for 2 workers: base + 2*per_worker
        ram = RAM_BASE_GB + 2 * RAM_PER_WORKER_GB
        result = compute_workers('auto', available_ram_gb=ram)
        assert result == 2


class TestExitCodes:
    """Test that the app exits with appropriate codes."""

    def _run_app(self, *args):
        """Run app.py as a subprocess and return the result."""
        cmd = [sys.executable, '-m', 'src.app'] + list(args)
        return subprocess.run(
            cmd, capture_output=True, text=True,
            cwd='/workspaces/perch-runner',
        )

    def test_missing_source_exits_nonzero(self):
        result = self._run_app(
            '--embed', '--source', '/nonexistent/path', '--output', '/tmp'
        )
        assert result.returncode != 0

    def test_no_action_exits_zero(self):
        # No --embed or --classify with valid paths = nothing to do, should exit 0
        result = self._run_app('--source', '/tmp', '--output', '/tmp')
        assert result.returncode == 0

    def test_invalid_model_exits_nonzero(self):
        result = self._run_app(
            '--embed', '--model_choice', 'bogus_model'
        )
        assert result.returncode != 0
