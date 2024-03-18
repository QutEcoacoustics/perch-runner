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

if [[ ! -e "$source" ]]; then
    echo "Error: Source path does not exist: $source"
    exit 1
elif [[ -d "$source" && -z "$(ls -A "$source")" ]]; then
    echo "Error: Source directory is empty: $source"
    exit 1
fi

if [[ ! -d "$output" ]]; then
    echo "Error: Output folder does not exist: $output"
    exit 1
fi

# based on a recognizer name, get the name of the default config file for that recognizer
declare -A recognizer_configs
recognizer_configs["pw"]="pw.classify.yml"
recognizer_configs["cgw"]="cgw.classify.yml"

if [[ -n ${recognizer_configs[$recognizer]} ]]; then
    echo "Using config file: ${recognizer_configs[$recognizer]}"
else
    echo "Recognizer $key not supported"
    exit 1
fi

# paths to things inside the container, to be mounted
embeddings_container="/mnt/embeddings"
output_container="/mnt/output"
output_dir=$output_container/search_results

command="python /app/src/app.py classify --source_folder $embeddings_container --output_folder $output_container  --config_file ${recognizer_configs[$recognizer]}"

#command="python /app/src/app.py --embeddings_dir $embeddings_container --model_path $model_path --output_dir $output_dir --skip_if_file_exists"



 
echo "launching container with command: $command"

set -x
docker run --user appuser:appuser --rm \
-v "$(pwd)/src":/app/src \
-v "$source":$embeddings_container \
-v "$output":$output_container $image $command
set +x
