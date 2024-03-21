# Step 1. Install Docker

Go to https://www.docker.com/get-started/ and install Docker for your computer if you don't already have it installed. The recognizer is provided as a docker container, and you need this software installed to be able to run it. 

# Step 2. Download the analysis script

Right click on the link and choose "save link as"
  - Windows: <a href="https://raw.githubusercontent.com/QutEcoacoustics/perch-runner/main/scripts/analyze.ps1" download>analyze.ps1</a>
  - Linux / x86 Mac: <a href="https://raw.githubusercontent.com/QutEcoacoustics/main/scripts/analyze.sh" download>analyze.sh</a>

# Step 3. Embed audio

1. Open a terminal window
2. Change directory to the directory where your downloaded analysis script is
3. Run the following command:
  - windows: `powershell -ExecutionPolicy Bypass -File .\analyze.ps1 generate [path_to_audio_folder] [path_to_embeddings_output_folder]`
  - linux or intel mac: `./analyze.sh generate [path_to_audio_folder] [path_to_embeddings_output_folder]`

Notes
- In the command above, replace the placeholders with your real audio and output folder. The output folder is where the embeddings files will get saved.
- This will take quite a long time to run. It's possible that it's too slow to be practical, depending on how much audio you have

# Step 3. Classify embeddings

1. Open a terminal window
2. Change directory to this scripts directory
3. Run the following command:
  - windows: `powershell -ExecutionPolicy Bypass -File analyze.ps1 classify [path_to_audio_folder] [path_to_embeddings_folder] 'pw'`
  - linux or intel mac: `./analyze.sh classify [path_to_audio_folder] [path_to_classifications_output_folder] 'pw'`


Notes
- In the command above, replace the placeholders with your real embeddings folder (which you specified in step 1) and output folder. The output folder is where the csv files of classifications will be saved. These files score each 5 second segment. Any score above zero is a positive classification. 
