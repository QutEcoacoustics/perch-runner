#!/usr/bin/env bash

# Default settings
PUSH=false
ACTION="--load"
NO_CACHE=""
VERSION=""

# Parse arguments
for arg in "$@"; do
  case "$arg" in
    --push)
      PUSH=true
      PLATFORMS="linux/amd64,linux/arm64"
      ACTION="--push"
      ;;
    --no-cache)
      NO_CACHE="--no-cache"
      ;;
    *)
      if [ -z "$VERSION" ]; then
        # First non-flag argument is the version
        VERSION="$arg"
      else
        echo "Unknown argument: $arg"
        echo "Usage: $0 [--push] [--no-cache] [version]"
        exit 1
      fi
      ;;
  esac
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