#docker image list

# tests are not included in image, but pytest is installed. 
# To run tests in the container, we mount the test directory

# run tests from within the container
set -x
docker run \
-v $(pwd)/tests/:/app/tests \
--entrypoint python \
qutecoacoustics/perchrunner:latest -m pytest /app/tests/app_tests
set +x

