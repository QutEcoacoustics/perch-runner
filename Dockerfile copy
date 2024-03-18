FROM --platform=linux/amd64 python:3.10-bookworm as perch_runner

RUN apt update 
RUN apt install -y libsndfile1 ffmpeg

# download and install poetry
# consider changing to pip install 'poetry==$POETRY_VERSION'
# see https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
RUN curl -sSL https://install.python-poetry.org | python3 -

RUN mkdir /app && mkdir /app/src
WORKDIR /app

# download and unzip Perch
ARG perch_repo=https://github.com/google-research/chirp/archive/refs/heads/main.zip
RUN wget $perch_repo && unzip main.zip && rm main.zip
# WORKDIR /app/perch-main
WORKDIR /app

# we use our own pyproject file modified from the perch on
# because we have extra dev dependencies, and also we remove some unecessary
# deps from the basic perch
COPY ./pyproject.toml /app

# install perch dependencies (not in venv since we are using docker)
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false --local
# RUN /root/.local/bin/poetry install
# this is due to "connection pool is full" error
# https://stackoverflow.com/questions/74385209/poetry-install-throws-connection-pool-is-full-discarding-connection-pypi-org
RUN poetry config installer.max-workers 10
RUN poetry install --no-interaction --no-ansi -vvv

# install perch_runner dependencies

COPY ./src /app/src

RUN python /app/src/download_model.py --version 4 --destination /models

ENV PYTHONPATH "${PYTHONPATH}:/app/perch-main"

