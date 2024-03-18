FROM --platform=linux/amd64 python:3.10-bookworm as perch_runner_dev

RUN apt update && apt install -y libsndfile1 ffmpeg

RUN mkdir /app && mkdir /app/src
WORKDIR /app

COPY ./pyproject.toml /app

# RUN pip install git+https://github.com/google-research/perch.git@8cc4468afaac730e77d84ac447f0874f09d10a25
RUN pip install git+https://github.com/google-research/perch.git@3746672d406c6cfe48acb0e725248cea05f57445

COPY ./src /app/src

# this is the trained linear models
COPY ./models /models

# this is the embedding model
RUN python /app/src/download_model.py --version 4 --destination /models

RUN pip install librosa numpy pytest pytest-mock

RUN useradd -u 1000 -ms /bin/bash appuser

