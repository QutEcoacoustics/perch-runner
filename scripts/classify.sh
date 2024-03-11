#! /bin/bash

# launches a docker container interactive with the necessary mounts for running inference on a folder of embeddings

# Argument Parsing
source="$1"
output="$2"
recognizer="$3"
image="${4:-qutecoacoustics/perchrunner:latest}"

# Required Parameter Validation
if [[ -z "$source" || -z "$output" || -z "$recognizer" ]]; then
    echo "Error: Missing required parameters (--source, --output, --recognizer)"
    exit 1 
fi

# Required Parameter Validation
if [[ -z "$source" || -z "$output" || -z "$recognizer" ]]; then
    echo "Error: Missing required parameters (source, output, recognizer)"
    exit 1 
fi

# Source and Output Path Checks
if [[ ! -d "$source" ]]; then
    echo "Error: Source audio folder does not exist: $source"
    exit 1
fi

if [[ ! -d "$output" ]]; then
    echo "Error: Output folder does not exist: $output"
    exit 1
fi

if [[ ! -s "$source" ]]; then
    echo "Error: Source is empty: $source"
    exit 1
fi

# paths to things inside the container, to be mounted
embeddings_container="/mnt/embeddings"
output_container="/mnt/output"
output_dir=$output_container/search_results

# paths to things inside the container, existing in the image (not mounted)
# NOTE: unsanitized, trusted input only
model_path="/models/$recognizer"

command="python /app/src/inference_parquet.py --embeddings_dir $embeddings_container --model_path $model_path --output_dir $output_dir --skip_if_file_exists"
 
echo "launching container with command: $command"

docker run --user appuser:appuser --rm \
# -v "$(pwd)/src":/app/src \
-v "$embeddings_host":$embeddings_container \
-v "$output_host":$output_container $image $command

