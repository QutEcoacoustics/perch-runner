#!/usr/bin/env bash
# Runs app_tests and integration tests inside the built container. This file is called from the host.
# Network is guaranteed blocked at the Docker level (--network=none).
# Usage: ./run_tests_in_container.sh [extra pytest args...]
#
# Examples:
#   ./run_tests_in_container.sh
#   ./run_tests_in_container.sh -k "test_full_pipeline"
#   ./run_tests_in_container.sh --tb=long

set -euo pipefail

IMAGE="${IMAGE:-qutecoacoustics/perchrunner:latest}"

echo "Using Docker image: $IMAGE"

exec docker run --rm --network=none \
  --entrypoint /app/tests/run_tests \
  "$IMAGE" \
  "$@"
