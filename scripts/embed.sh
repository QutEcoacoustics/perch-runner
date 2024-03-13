#! /bin/bash

# launches a docker container interactive with the necessary mounts for running inference on a folder of embeddings

# Argument Parsing
source="$1"
output="$2"
image="${3:-qutecoacoustics/perchrunner:latest}"

# Required Parameter Validation
if [[ -z "$source" || -z "$output" ]]; then
    echo "Error: Missing required parameters (source, output)"
    exit 1 
fi

echo $(pwd)

echo $source

# Source Path Checks
if [[ ! -s "$source" ]]; then
    echo "Error: Source audio folder does not exist: $source"
    exit 1
fi

if [[ ! -s "$source" ]]; then
    echo "Error: Source is empty: $source"
    exit 1
fi

# paths to things inside the container, to be mounted
source_container="/mnt/input"
output_container="/mnt/output"

source_folder_host=$(dirname "$source")
source_basename=$(basename "$source")

command="python /app/src/app.py generate --source_file $source_container/$source_basename --output_folder $output_container"
 
echo "launching container with command: $command"

set -x
docker run --user appuser:appuser --rm \
-v "$(pwd)/src":/app/src \
-v "$source_folder_host":$source_container \
-v "$output":$output_container $image $command
set +x

# add this in to mount the source directory to run changes without rebuilding
# -v "$(pwd)/src":/app/src \