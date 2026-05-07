FROM python:3.12-slim AS base

# tells uv to install packages globally instead of in a venv, since we're in a container
ENV UV_SYSTEM_PYTHON=1

RUN apt update && apt install -y libsndfile1 ffmpeg

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install 'perch-hoplite[tf]' pytest

# --- Test Stage: runs tests and downloads models into kagglehub cache ---
FROM base AS test
ARG DEV=false
WORKDIR /app
COPY . .
# Ensure the cache directory exists so COPY in final stage always succeeds
RUN mkdir -p /root/.cache/kagglehub
# Only run allow_network tests in non-dev mode — these download and cache models.
# Full test suite is run post-build via run_tests_in_container.sh or CI.
RUN if [ "$DEV" != "true" ]; then \
    touch /.dockerenv && \
    python -m pytest tests/app_tests tests/integration -v -m "allow_network"; \
    fi

# --- Final Stage ---
FROM base AS final
WORKDIR /app
COPY . .
COPY --from=test /root/.cache/kagglehub /root/.cache/kagglehub
ENTRYPOINT ["python", "-m", "src.app"]
