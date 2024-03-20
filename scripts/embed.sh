#! /bin/bash

# launches a docker container interactive with the necessary mounts for generating embeddings on a folder of wav files
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


# paths to things inside the container, to be mounted
source_container="/mnt/input"
output_container="/mnt/output"

source_folder_host=$(dirname "$source")
source_basename=$(basename "$source")

command="python /app/src/app.py generate --source $source_container/$source_basename --output $output_container"
 
echo "launching container with command: $command"

# Convert to absolute paths
absolute_source=$(realpath "$source_folder_host")
absolute_output=$(realpath "$output")

set -x
docker run --user appuser:appuser --rm \
-v "$absolute_source":$source_container \
-v "$absolute_output":$output_container $image $command
set +x

# add this in to mount the source directory to run changes without rebuilding
# -v "$(pwd)/src":/app/src \