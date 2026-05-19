FROM python:3.12-slim AS base

# tells uv to install packages globally instead of in a venv, since we're in a container
ENV UV_SYSTEM_PYTHON=1

RUN apt update && apt install -y libsndfile1 ffmpeg

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG PERCH_HOPLITE_VERSION=1.0.1
RUN uv pip install "perch-hoplite[tf]==${PERCH_HOPLITE_VERSION}" pytest pyarrow

# --- Models Stage: resolve presets, generate models.json, download models ---
FROM base AS models
WORKDIR /app
COPY src/__init__.py src/__init__.py
COPY src/download_models.py src/download_models.py
RUN python -m src.download_models

# --- Final Stage ---
FROM base AS final
WORKDIR /app
COPY --from=models /root/.cache/kagglehub /root/.cache/kagglehub
COPY . .
COPY --from=models /app/src/models.json src/models.json
ARG VERSION=dev
ENV APP_VERSION=${VERSION}
ENTRYPOINT ["python", "-m", "src.app"]
