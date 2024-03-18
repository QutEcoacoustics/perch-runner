FROM --platform=linux/amd64 python:3.10-bookworm as perch_runner_dev

RUN apt update 
RUN apt install -y libsndfile1 ffmpeg

# download and install poetry
# consider changing to pip install 'poetry==$POETRY_VERSION'
# see https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# RUN curl -sSL https://install.python-poetry.org | python3 -

RUN mkdir /app && mkdir /app/src
WORKDIR /app


# we use our own pyproject file modified from the perch on
# because we have extra dev dependencies, and also we remove some unecessary
# deps from the basic perch
COPY ./pyproject.toml /app

# install perch dependencies (not in venv since we are using docker)
# ENV PATH="/root/.local/bin:$PATH"
# RUN poetry config virtualenvs.create false --local
# RUN /root/.local/bin/poetry install
# # this is due to "connection pool is full" error
# # https://stackoverflow.com/questions/74385209/poetry-install-throws-connection-pool-is-full-discarding-connection-pypi-org
# RUN poetry config installer.max-workers 10
# RUN poetry install --no-interaction --no-ansi -vvv

# install perch_runner dependencies

# RUN pip install git+https://github.com/google-research/perch.git@8cc4468afaac730e77d84ac447f0874f09d10a25
RUN pip install git+https://github.com/google-research/perch.git@3746672d406c6cfe48acb0e725248cea05f57445

# there seems to be a problem with the way flax interacts with jaxlib 
# this is a temporary fix to see if it helps without introducing other problems
# RUN pip install --upgrade flax jax jaxlib

#COPY --from=perch_runner_build /app/requirements.txt /app

# RUN pip install .
# RUN pip install .[dev]

COPY ./src /app/src

# this is the trained linear models
COPY ./models /models

# this is the embedding model
RUN python /app/src/download_model.py --version 4 --destination /models

# ENV PYTHONPATH "${PYTHONPATH}:/app/perch-main"

#COPY ./tests /app/tests

RUN pip install librosa
RUN pip install numpy 
RUN pip install pytest pytest-mock

RUN useradd -u 1000 -ms /bin/bash appuser

