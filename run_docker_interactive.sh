#! /bin/bash

# launches a docker container interactive with the necessary mounts for embedding

port=${1:-8888}
image=${2:-ch3}

unlabelled_local="/mnt/availae/Phil/Australian Wildlife Conservancy/Richard Seaton/napco_survey_project"
labelled_local=/mnt/c/Users/Administrator/Documents/phil
output_local=/mnt/c/Users/Administrator/Documents/phil/output
docker run  --gpus all --user root:root -ti -p $port:$port \
-v "$(pwd)":/app/scripts \
-v "$unlabelled_local":/napco_survey_project_audio \
-v "$labelled_local":/phil \
-v "$output_local":/output $image bash

