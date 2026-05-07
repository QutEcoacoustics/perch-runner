#!/usr/bin/env bash

# Default settings
PUSH=false
PLATFORMS="linux/arm64" # Default to local Mac architecture for speed
ACTION="--load"

# Check for --push flag
for arg in "$@"; do
  if [ "$arg" == "--push" ]; then
    PUSH=true
    # When pushing, we build for BOTH Mac and Linux servers
    PLATFORMS="linux/amd64,linux/arm64"
    ACTION="--push"
  fi
done

# Versioning
PR_VERSION=$(date '+%Y%m%d%H%M%S')_$(git rev-parse --short HEAD)
tag=qutecoacoustics/perchrunner

echo "Mode: $( [ "$PUSH" = true ] && echo 'RELEASE (Pushing to Docker Hub)' || echo 'LOCAL (Testing on Mac)' )"
echo "Building version: $PR_VERSION"
echo "Platforms: $PLATFORMS"

# Buildx command
docker buildx build \
  --platform "$PLATFORMS" \
  -t $tag:$PR_VERSION \
  -t $tag:latest \
  $ACTION \
  --build-arg VERSION=$PR_VERSION \
  --progress=plain \
  .