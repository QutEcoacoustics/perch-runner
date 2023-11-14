FROM --platform=linux/amd64 python:3.10-bookworm as chirp_runner

RUN apt update 
RUN apt install -y libsndfile1 ffmpeg

# download and install poetry
# consider changing to pip install 'poetry==$POETRY_VERSION'
# see https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
RUN curl -sSL https://install.python-poetry.org | python3 -

RUN mkdir /app && mkdir /app/scripts
WORKDIR /app

# download and unzip chirp
ARG chirp_repo=https://github.com/google-research/chirp/archive/refs/heads/main.zip
RUN wget $chirp_repo && unzip main.zip && rm main.zip
WORKDIR /app/perch-main

# install chirp dependencies (not in venv since we are using docker)
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false --local
# RUN /root/.local/bin/poetry install
# this is due to "connection pool is full" error
# https://stackoverflow.com/questions/74385209/poetry-install-throws-connection-pool-is-full-discarding-connection-pypi-org
RUN poetry config installer.max-workers 10
RUN poetry install --no-interaction --no-ansi -vvv

COPY launch_notebook.sh run_embed.sh train_linear_model.py  /app/scripts/
