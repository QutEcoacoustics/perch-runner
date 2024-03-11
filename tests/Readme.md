This repository contains several parts
- The App itself
- Dockerfile for building the environment to run the App
- Scripts to run from the docker host to launch the docker container. 

We have two sets of tests. 

- App Tests: 
  - For testing the app iself
  - These ensure the core logic and functionality of the application work as intended within its environment, which in most cases will be the docker container.
- Docker Tests: 
  - For testing the scripts that run the docker container
  - These Verify that from the host the app is launched, files/directories are mounted, and the overall deployment process works without unexpected issues.

There are some fixtures that are shared by both sets of tests, and both sets of tests make use of the input and output folders and source files.  

- Input
  - This is where we tell the app to find files to work on.

- files
  - This contains any test files that the app will work on. They are copied to the input folder before each test

- Output
  - This is where we tell the app to write results files
  - When running app-tests, this must be mounted to a specific location that the app expects

The input and output folders
- When running app-tests in the container, these must be mounted to a specific that the app expects input
- When running docker-tests, this is passed to the scripts that launch the container, which mount them to the location that the app expects
- These folders are both cleaned up after each test