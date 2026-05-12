FROM python:3.12-slim AS base

# tells uv to install packages globally instead of in a venv, since we're in a container
ENV UV_SYSTEM_PYTHON=1

RUN apt update && apt install -y libsndfile1 ffmpeg

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install 'perch-hoplite[tf]' pytest

# --- Models Stage: resolve presets, generate models.json, download models ---
FROM base AS models
WORKDIR /app
COPY src/__init__.py src/__init__.py
COPY src/download_models.py src/download_models.py
RUN python -m src.download_models

# --- Test Stage: runs tests using cached models ---
FROM base AS test
ARG DEV=false
WORKDIR /app
COPY --from=models /root/.cache/kagglehub /root/.cache/kagglehub
COPY . .
# Run model tests during build to verify cached models work.
# Full test suite is run post-build via run_tests_in_container.sh or CI.
RUN if [ "$DEV" != "true" ]; then \
    touch /.dockerenv && \
    python -m pytest tests/app_tests/test_embed_models.py -v; \
    fi

# --- Final Stage ---
FROM base AS final
WORKDIR /app
COPY --from=models /root/.cache/kagglehub /root/.cache/kagglehub
COPY . .
COPY --from=models /app/src/models.json src/models.json
ARG VERSION=dev
ENV APP_VERSION=${VERSION}
ENTRYPOINT ["python", "-m", "src.app"]
