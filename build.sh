#!/usr/bin/env bash

# Default settings
PUSH=false
QUICK=false
PLATFORMS="linux/arm64" # Default to local Mac architecture for speed
ACTION="--load"
NO_CACHE=""
TARGET=""

# Check for flags
for arg in "$@"; do
  if [ "$arg" == "--push" ]; then
    PUSH=true
    # When pushing, we build for BOTH Mac and Linux servers
    PLATFORMS="linux/amd64,linux/arm64"
    ACTION="--push"
  fi
  if [ "$arg" == "--no-cache" ]; then
    NO_CACHE="--no-cache"
  fi
  if [ "$arg" == "--quick" ]; then
    QUICK=true
    TARGET="--target final"
  fi
done

# Versioning
PR_VERSION=$(date '+%Y%m%d%H%M%S')_$(git rev-parse --short HEAD)
tag=qutecoacoustics/perchrunner

if [ "$QUICK" = true ]; then
  echo "Mode: QUICK (skipping tests, code-only rebuild)"
else
  echo "Mode: $( [ "$PUSH" = true ] && echo 'RELEASE (Pushing to Docker Hub)' || echo 'LOCAL (Testing on Mac)' )"
fi
echo "Building version: $PR_VERSION"
echo "Platforms: $PLATFORMS"

# Buildx command
docker buildx build \
  --platform "$PLATFORMS" \
  -t $tag:$PR_VERSION \
  -t $tag:latest \
  $ACTION \
  $NO_CACHE \
  $TARGET \
  --build-arg VERSION=$PR_VERSION \
  --progress=plain \
  .