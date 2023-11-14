#! /bin/bash

# runs embed the embed script for one site of the napco survey

input_base=${2:-""}
output_base=${3:-"/output"}

# run on big data without docker
# ./run_embed.sh 050 "/mnt/availae/Phil/Australian Wildlife Conservancy/Richard Seaton/" /mnt/c/Users/Administrator/Documents/phil/output

input="$input_base/napco_survey_project_audio/PW/inference_datasets/20230413/$1/*/*.wav"
output="$output_base/site_$1"

poetry run python chirp/inference/embed_audio.py "$input" "$output" "$model"