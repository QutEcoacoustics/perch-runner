# perch-runner

A Docker container that generates [Perch](https://github.com/google-research/perch) audio embeddings for folders of audio files.

## Quick Start

```bash
docker run --rm \
  -v /path/to/audio:/mnt/input \
  -v /path/to/output:/mnt/output \
  qutecoacoustics/perchrunner:latest analyze --embed
```

This processes all audio files in `/path/to/audio` and writes Parquet embedding files to `/path/to/output/embeddings/`.

## Usage

```
docker run --rm \
  -v <source>:/mnt/input \
  -v <output>:/mnt/output \
  [-v <config_dir>:/mnt/config] \
  qutecoacoustics/perchrunner:latest <subcommand> [options]
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `analyze` | Run embedding/classification pipeline |
| `version` | Print perch-runner, perch-hoplite, and model versions |
| `config` | Print default resolved config as JSON |

### Analyze Options

| Flag | Description | Default |
|------|-------------|---------|
| `--embed [format]` | Generate embeddings. Examples: `parquet`, `csv`, `parquet-columns`, `csv-columns` | `parquet` |
| `--classify [format]` | Classification mode selector (accepted values: `parquet`, `csv`, `hoplite`; currently not implemented) | `csv` |
| `--model_choice` | Model to use: `perch_v2` or `perch_8` | `perch_v2` |
| `--embedding_table_format` | Table layout: `serialized` or `columns` | `serialized` |
| `--embeddings_output_path_template` | Output path template tokens: `{parents}`, `{basename}`, `{ext}`, `{embedding_table_format}`, `{analysis}` | `{parents}/{basename}/embeddings{ext}` |
| `--embeddings_output_path_type` | Preset output layout: `flat_basename`, `nested_basename`, `nested`, `flat` | None |
| `--db_path` | Database path; relative paths resolve under output | `db` |
| `--file_glob` | Glob pattern for audio files, e.g. `*/*`, `*/*/*` | Auto-detected |
| `--workers` | Worker count or `auto` | `auto` |
| `--log_level` | App log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `--hoplite_log_level` | Library/root log level | `WARNING` |
| `--tf_log_level` | TensorFlow C++ log level | `WARNING` |
| `--log_file` | Optional log file path | None |
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
  qutecoacoustics/perchrunner:latest analyze --config_file /mnt/config/config.yml
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

