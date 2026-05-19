Test layout overview

- `tests/unit_tests/`
  - Unit tests for single functions/modules.
  - Typically fast and mock-heavy.

- `tests/integration_tests/`
  - Multi-step integration tests for application modules (`src/*`).
  - Includes component-level tests with real side effects (for example, writing outputs).

- `tests/end_to_end_tests/`
  - End-to-end CLI tests.
  - These invoke the app via subprocess and (from host) via `docker run`.

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
  - Runs end-to-end tests only (`tests/end_to_end_tests`).

- Against built image from host (inside container):
  - `./run_tests_in_container.sh`
  - Runs unit, integration, and end-to-end tests inside the built image.