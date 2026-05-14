import os
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURES_DIR = Path("tests/files")


def _in_container() -> bool:
    """Return True if running inside a Docker container."""
    return Path("/.dockerenv").exists() or os.environ.get("PERCH_ENV") == "container"


@pytest.fixture
def workspace(tmp_path):
    """Provides source/output/config dirs in a temp location (auto-cleaned by pytest)."""
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    return source, output, config


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
            sys.executable, "src/app.py",
            "--source", str(source),
            "--output", str(output),
        ]
        if config_file:
            cmd += ["--config_file", str(config_file)]
        cmd += list(extra_args)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path.cwd())

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}"
            )
        return result
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

        command = ["docker", "run", "--rm", "--network=none"] + mounts + [image] + cmd_args
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker command failed (exit {result.returncode}):\n"
                f"  cmd: {' '.join(command)}\n"
                f"  stdout: {result.stdout}\n"
                f"  stderr: {result.stderr}"
            )
        return result
    return _run
