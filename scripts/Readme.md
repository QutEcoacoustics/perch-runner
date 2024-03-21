# Step 1. Install Docker

Go to https://www.docker.com/get-started/ and install Docker for your computer if you don't already have it installed. The recognizer is provided as a docker container, and you need this software installed to be able to run it. 

# Step 2. Embed audio

1. Download the embed script from this directory
  - Windows: [embed.ps1](https://raw.githubusercontent.com/QutEcoacoustics/perch-runner/docker-launch-scripts-and-tests/scripts/embed.ps1)
  - Linux / x86 Mac: [embed.sh](https://raw.githubusercontent.com/QutEcoacoustics/perch-runner/docker-launch-scripts-and-tests/scripts/embed.sh)
2. Open a terminal window
2. Change directory to this scripts directory
3. Run the following command:
  - windows: `pwsh embed.ps1 [path_to_audio_folder] [path_to_embeddings_output_folder]`
  - linux or intel mac: `./embed.sh [path_to_audio_folder] [path_to_embeddings_output_folder]`


Notes
- In the command above, replace the placeholders with your real audio and output folder. The output folder is where the embeddings files will get saved.
- This will take quite a long time to run. It's possible that it's too slow to be practical, depending on how much audio you have

# Step 3. Classify embeddings

1. Download the classify script from this directory
  - Windows: [classify.ps1](https://raw.githubusercontent.com/QutEcoacoustics/perch-runner/docker-launch-scripts-and-tests/scripts/classify.ps1)
  - Linux / x86 Mac: [classify.sh](https://raw.githubusercontent.com/QutEcoacoustics/perch-runner/docker-launch-scripts-and-tests/scripts/classify.sh)
2. Open a terminal window
2. Change directory to this scripts directory
3. Run the following command:
  - windows: `pwsh classify.ps1 [path_to_audio_folder] [path_to_embeddings_folder] 'pw'`
  - linux or intel mac: `./classify.sh [path_to_audio_folder] [path_to_classifications_output_folder] 'pw'`


Notes
- In the command above, replace the placeholders with your real embeddings folder (which you specified in step 1) and output folder. The output folder is where the csv files of classifications will be saved. These files score each 5 second segment. Any score above zero is a positive classification. 
