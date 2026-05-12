Test layout overview

- `tests/app_tests/`
  - Unit and integration-style tests for application modules (`src/*`).
  - Includes fast mocked tests and slower real-model integration tests.

- `tests/integration/`
  - End-to-end CLI tests.
  - These invoke the container via `docker run` from host-side pytest.

- `tests/files/`
  - Shared fixture assets (audio samples, fixture embeddings, configs).

- `tests/shared_fixtures/`
  - Shared pytest fixtures/helpers used across test groups.

Network policy

- All tests run with network blocked by default via `tests/conftest.py`.
- Models must be pre-cached in the image/build environment.

Running tests

- In dev container:
  - `pytest`

- Against built image from host:
  - `./run_tests.sh`

- From host, run full suite inside built image:
  - `./run_tests_in_container.sh`