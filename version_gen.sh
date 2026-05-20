#!/usr/bin/env bash
# Outputs a version string: v2_<timestamp>_<git short hash>
# this is used by build.sh as a default version, and by build.yml to set the version in CI
# It's not part of the container. 

# If you want a different prefix, change here:
VERSION_PREFIX="v2_"

# Get timestamp and git hash
TIMESTAMP=$(date '+%Y%m%d%H%M%S')
GIT_HASH=$(git rev-parse --short HEAD)

# Print version string
printf "%s%s_%s\n" "$VERSION_PREFIX" "$TIMESTAMP" "$GIT_HASH"
