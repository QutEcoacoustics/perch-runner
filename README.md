# perch-runner

A Docker container that generates [Perch](https://github.com/google-research/perch) audio embeddings for folders of audio files.

## Quick Start

```bash
docker run --rm \
  -v /path/to/audio:/mnt/input \
  -v /path/to/output:/mnt/output \
  qutecoacoustics/perchrunner:latest --embed
```

This processes all audio files in `/path/to/audio` and writes Parquet embedding files to `/path/to/output/embeddings/`.

## Usage

```
docker run --rm \
  -v <source>:/mnt/input \
  -v <output>:/mnt/output \
  [-v <config_dir>:/mnt/config] \
  qutecoacoustics/perchrunner:latest [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--embed [format]` | Generate embeddings. Common formats: `parquet` or `hoplite` | `parquet` |
| `--model_choice` | Model to use: `perch_v2` or `perch_8` | `perch_v2` |
| `--embedding_table_format` | Table layout: `serialized` or `columns` | `serialized` |
| `--file_glob` | Glob pattern for audio files, e.g. `*/*`, `*/*/*` | Auto-detected |
| `--config_file` | Path to a YAML config file | None |
| `--source` | Override source path (default: `/mnt/input`) | `/mnt/input` |
| `--output` | Override output path (default: `/mnt/output`) | `/mnt/output` |

### Supported Audio Formats

`.wav`, `.flac`, `.mp3`, `.ogg`

### Output Structure

Audio files are discovered relative to the source directory. Output mirrors that structure:

```
/mnt/output/
  embeddings/
    site1/
      recording.wav/
        embeddings.parquet
    site2/
      another.flac/
        embeddings.parquet
```

Each Parquet file contains one row per 5-second window with columns: `source`, `channel`, `offset`, `embeddings` (serialized numpy array).

With `--embedding_table_format columns`, the `embeddings` column is replaced by individual dimension columns (`f0000`, `f0001`, ...).

With `--embed hoplite`, the raw Hoplite/USearch database is kept at `/mnt/output/hoplite/` and no Parquet files are produced.

### Config File

Instead of CLI flags, you can mount a YAML config file:

```yaml
source: /mnt/input
output: /mnt/output
embed: parquet
model_choice: perch_v2
embedding_table_format: serialized
file_glob: "*/*"
```

```bash
docker run --rm \
  -v /path/to/audio:/mnt/input \
  -v /path/to/output:/mnt/output \
  -v /path/to/config:/mnt/config \
  qutecoacoustics/perchrunner:latest --config_file /mnt/config/config.yml
```

## Models

| Model | Embedding Dimensions | Description |
|-------|---------------------|-------------|
| `perch_v2` | 1536 | Default. Google Perch v2 bird embedding model |
| `perch_8` | 1280 | Google Perch v8 |

Models are cached in the Docker image at build time — no internet access is required at runtime.

## Building

```bash
# Local build (current architecture)
./build.sh

# Local build with explicit tag/version (for example, dev)
./build.sh dev

# Build and push to Docker Hub (amd64 + arm64)
./build.sh --push
```

The image build resolves/downloads models and embeds them in the image cache.
Tests are run after build in CI and during local development.

## Testing

### Inside the dev container (development)

```bash
# Run all tests (network is blocked; models must be cached)
pytest
```

### From the host against a built image

```bash
# run the full suite of tests in the container from the host
./run_tests_in_container.sh
```

```bash
# run integration tests only, on the host
./run_tests.sh
```

This runs host integration tests in `tests/integration`, which execute the built
container via `docker run` and validate produced outputs.  Requires pytest to be
installed on the host environment. 


## License

Apache 2.0

