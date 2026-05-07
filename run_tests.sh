#!/usr/bin/env bash
# Run integration tests from the host against a built container.
# pytest runs on the host; the runner fixture invokes `docker run` with
# tmp_path mounted, then asserts on the output files.
#
# Prerequisites: pip install -r requirements-host.txt (just pytest)
# Requires: a built image (qutecoacoustics/perchrunner:latest)

set -euo pipefail

IMAGE="${IMAGE:-qutecoacoustics/perchrunner:latest}"
export IMAGE

python -m pytest tests/integration -v "$@"
