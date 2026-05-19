import os
import subprocess
import sys
from pathlib import Path

import pytest

# This conftest.py provides fixtures for end-to-end tests of the CLI.
# depending on the 
_CONTAINER_TEST_FILES = Path("/app/tests/files")
_HOST_TEST_FILES = Path(__file__).parents[2] / "tests" / "files"
CANONICAL_TEST_FILES = _CONTAINER_TEST_FILES if _CONTAINER_TEST_FILES.exists() else _HOST_TEST_FILES


def _run_command(cmd, description="Command", **subprocess_kwargs):
    """Run a command via subprocess and check for errors.
    
    Args:
        cmd: List of command arguments to run
        description: Prefix for error messages
        **subprocess_kwargs: Additional kwargs to pass to subprocess.run()
    
    Returns:
        CompletedProcess result
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        **subprocess_kwargs
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(str(x) for x in cmd)}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
    return result


def _in_container() -> bool:
    """Return True if running inside a Docker container."""
    return Path("/.dockerenv").exists() or os.environ.get("PERCH_ENV") == "container"


@pytest.fixture
def workspace(tmp_path):
    """
    Provides fresh input/output/config dirs for each test, copying canonical test files from /app/tests/files.
    - source: temp input dir (test can copy needed files from CANONICAL_TEST_FILES)
    - output: temp output dir
    - config: temp config dir
    """
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    return source, output, config, CANONICAL_TEST_FILES


@pytest.fixture
def runner():
    """Returns a callable that runs the app either directly or via Docker,
    depending on whether we're inside a container."""

    if _in_container():
        return _subprocess_runner()
    else:
        return _docker_runner()


def _subprocess_runner():
    """Runner that invokes src/app.py as a local subprocess."""
    def _run(source, output, *extra_args, config_file=None):
        cmd = [
            sys.executable, "src/app.py", "analyze",
            "--source", str(source),
            "--output", str(output),
        ]
        if config_file:
            cmd += ["--config_file", str(config_file)]
        cmd += list(extra_args)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path.cwd())

        return _run_command(
            cmd,
            description="Subprocess command",
            cwd=str(Path.cwd()),
            env=env,
        )
    return _run


def _docker_runner():
    """Runner that invokes the app inside a Docker container.
    Network is guaranteed blocked at the Docker level (--network=none).
    """
    image = os.environ.get("IMAGE", "qutecoacoustics/perchrunner:latest")

    def _run(source, output, *extra_args, config_file=None):
        mounts = [
            "-v", f"{Path(source).absolute()}:/mnt/input",
            "-v", f"{Path(output).absolute()}:/mnt/output",
        ]
        cmd_args = ["--source", "/mnt/input", "--output", "/mnt/output"]

        if config_file:
            config_path = Path(config_file).absolute()
            config_dir = config_path.parent
            mounts += ["-v", f"{config_dir}:/mnt/config"]
            cmd_args += ["--config_file", f"/mnt/config/{config_path.name}"]

        cmd_args += list(extra_args)

        command = ["docker", "run", "--rm", "--network=none"] + mounts + [image, "analyze"] + cmd_args
        return _run_command(
            command,
            description="Docker command",
        )
    return _run