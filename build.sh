#!/usr/bin/env bash

# Default settings
PUSH=false
ACTION="--load"
NO_CACHE=""
VERSION=""

# Parse arguments
for arg in "$@"; do
  if [ "$arg" == "--push" ]; then
    PUSH=true
    PLATFORMS="linux/amd64,linux/arm64"
    ACTION="--push"
  elif [ "$arg" == "--no-cache" ]; then
    NO_CACHE="--no-cache"
  elif [ -z "$VERSION" ]; then
    # First non-flag argument is the version
    VERSION="$arg"
  fi
done

# Default version to timestamp if not provided
VERSION=${VERSION:-V2_$(date '+%Y%m%d%H%M%S')_$(git rev-parse --short HEAD)}
tag=qutecoacoustics/perchrunner

echo "Mode: $( [ "$PUSH" = true ] && echo 'RELEASE' || echo 'BUILD' )"
echo "Version: $VERSION"
if [ "$PUSH" = true ]; then
  echo "Platforms: $PLATFORMS"
else
  echo "Platforms: host default"
fi

docker buildx build \
  ${PLATFORMS:+--platform "$PLATFORMS"} \
  -t $tag:$VERSION \
  -t $tag:latest \
  $ACTION \
  $NO_CACHE \
  --build-arg VERSION=$VERSION \
  --progress=plain \
  .