#!/usr/bin/env bash
# Run end-to-end tests from the host against a built container.
# pytest runs on the host; the runner fixture invokes `docker run --network=none` with
# tmp_path mounted, ensuring network is blocked at the Docker level. Asserts on output files.
#
# Prerequisites: pip install -r requirements-host.txt
# Requires: a built image (qutecoacoustics/perchrunner:latest)

set -euo pipefail

IMAGE="${IMAGE:-qutecoacoustics/perchrunner:latest}"
export IMAGE

echo "Using Docker image: $IMAGE"

python -m pytest tests/end_to_end_tests -v "$@"
