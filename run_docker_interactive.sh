#! /bin/bash

# launches a docker container interactive with the necessary mounts for embedding

port=${1:-8888}
image=${2:-cr04}

unlabelled_local="/mnt/napco_survey_project_audio/PW"

labelled_local=/mnt/c/Users/Administrator/Documents/phil
output_local=/mnt/c/Users/Administrator/Documents/phil/output
docker run  --gpus all --user root:root -ti --rm -p $port:$port \
-v "$(pwd)/src":/app/src \
-v "$unlabelled_local":/mnt/pw \
-v "$labelled_local":/phil \
-v "$output_local":/output $image bash

