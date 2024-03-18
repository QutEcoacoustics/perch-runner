This document contains instructions to run one of the following recognizers over unlabelled audio
- pw (Plains Wanderer)
- cgw (Carpentarian Grass Wren)

# Setup

1. Prepare the paths to the relevant folders 

  You will need the following paths on your computer
  1. A folder with audio recordings to analyse (input files)
  2. A writable folder where we can store temporary files
  3. A writeable folder where we can save the results for each input file

2. Install docker if you don't have it.

You will need **Docker** installed on your computer

To install docker, please see `https://docs.docker.com/engine/install/`. Docker is software that can run a docker *container*. The recognizer (i.e. the trained model and the scripts needed to run it) and its dependencies are all bundled into this container. 


# Running the recognizer

Then, open a terminal shell and enter the following command

./Scripts/run_inference.sh 
