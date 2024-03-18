#!/usr/bin/env bash

#set -x

# crane version is the datetime and git hash
# this variable name is required by the github CI
PR_VERSION=$(date '+%Y%m%d%H%M%S')_$(git rev-parse --short HEAD)

echo "building container with version $PR_VERSION"

tag=qutecoacoustics/perchrunner

# use buildx build rather than just build, so that the --load is an option
# --load is required so that the image is saved to the local docker images when run on github CI

docker buildx build \
-t $tag:$PR_VERSION \
-t $tag:latest \
--load \
--build-arg VERSION=$PR_VERSION \
--progress=plain \
.