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

if [ -z "$VERSION" ]; then
  VERSION=$(bash "$(dirname "$0")/version_gen.sh")
fi
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