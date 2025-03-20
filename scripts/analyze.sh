#!/bin/bash

# Argument Parsing
analysis="$1"
source="$2"
output="$3"
recognizer="$4"
image="${5:-qutecoacoustics/perchrunner:latest}"
mount_src="${PERCH_RUNNER_MOUNT_SRC:-false}"

# Required Parameter Validation
if [[ -z "$analysis" || -z "$source" || -z "$output" ]]; then
    echo "Error: Missing required parameters (--analysis, --source, --output)"
    exit 1 
fi

# Additional validation for 'classify' analysis
if [[ "$analysis" == "classify" && -z "$recognizer" ]]; then
    echo "Error: Missing required parameter (--recognizer) for classify analysis"
    exit 1
fi

if [[ ! -e "$source" ]]; then
    echo "Error: Source path does not exist: $source"
    exit 1
elif [[ -d "$source" && -z "$(ls -A "$source")" ]]; then
    echo "Error: Source directory is empty: $source"
    exit 1
fi


# Initialize command with a default value
command=""

# if analysis is 'classify'
if [[ "$analysis" == "classify" ]]; then
    declare -A recognizer_configs
    recognizer_configs["pw"]="pw.classify.yml"
    recognizer_configs["cgw"]="cgw.classify.yml"
    recognizer_configs["mgw"]="mgw.classify.yml"

    # Check if the recognizer argument has been provided
    if [[ -n "$recognizer" ]]; then
        # Check if the provided recognizer is supported and set the config variable
        if [[ -n ${recognizer_configs[$recognizer]} ]]; then
            config=${recognizer_configs[$recognizer]}
            echo "Using config file: $config"
            config="--config $config"
        else
            echo "Recognizer $recognizer not supported"
            exit 1
        fi
    else
        # Set config variable to an empty string if no recognizer is provided
        config=""
    fi
fi

# Determine input container path
if [[ -f "$source" ]]; then
    source_base_name=$(basename "$source")
    input_container="/mnt/input/$source_base_name"
else
    input_container="/mnt/input"
fi

output_container="/mnt/output"

command="python /app/src/app.py $analysis --source $input_container --output $output_container $config"

echo "app command: $command"

# # Convert to absolute paths
absolute_host_source=$(realpath "$source")
absolute_host_output=$(realpath "$output")
 
mount_options="-v \"$absolute_host_source\":\"$input_container\" -v \"$absolute_host_output\":\"$output_container\""

# Add source directory mount if mount_src is true
echo "mount_src: $mount_src"
if [[ "$mount_src" == "true" ]]; then
    absolute_host_src=$(realpath "./src")
    mount_options="$mount_options -v \"$absolute_host_src:/app/src\""
fi

docker_command="docker run --user appuser:appuser --rm $mount_options $image $command"

echo "launching container with command: $docker_command"

# Use the mount_options variable in the docker run command
set -x
eval $docker_command
set +x
