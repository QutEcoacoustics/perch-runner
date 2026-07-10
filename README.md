# perch-runner

A Docker container that generates [Perch](https://github.com/google-research/perch) audio embeddings for folders of audio files.

## Quick Start

```bash
docker run --rm \
  -v /path/to/audio:/mnt/input \
  -v /path/to/output:/mnt/output \
  qutecoacoustics/perchrunner:latest analyze --embed
```

This processes all audio files in `/path/to/audio` and writes Parquet embedding files to `/path/to/output/`.

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
| `--embed` | Enable embedding export (boolean flag). Use `--embed` with no value to enable | `false` |
| `--embeddings_table_format` | Table layout: `serialized` or `columns` | `serialized` |
| `--embeddings_table_filetype` | File format for the embedding table: `parquet` or `csv` | `parquet` |
| `--embeddings_output_path_template` | Output path template for embedding files. Tokens: `{parents}`, `{basename}`, `{ext}`, `{embeddings_table_format}`, `{analysis}` | `{parents}/{basename}/{analysis}{ext}` |
| `--embeddings_output_path_type` | Preset output layout for embeddings: `flat_basename`, `nested_basename`, `nested`, `flat` | None |
| `--classify` | Enable Perch global classification output (boolean flag). Use `--classify` with no value to enable | `false` |
| `--classify_filetype` | File format for classification output: `parquet` or `csv` | `csv` |
| `--classify_output_path_template` | Output path template for classification files | None |
| `--classify_output_path_type` | Preset output layout for classification files: `flat_basename`, `nested_basename`, `nested`, `flat` | None |
| `--recognizers` | Path to a recognizers JSON file. Runs embeddings through linear classifiers and writes per-recognizer result files | None |
| `--recognizer_results_filetype` | File format for recognizer results: `parquet` or `csv` | `csv` |
| `--recognizer_output_path_template` | Output path template for recognizer result files. Tokens: `{recognizer_name}`, `{parents}`, `{basename}`, `{ext}`, `{analysis}` | `{recognizer_name}/{parents}/{basename}/{analysis}{ext}` |
| `--recognizer_output_path_type` | Preset output layout for recognizer results: `flat_basename`, `nested_basename`, `nested`, `flat` | None |
| `--model_choice` | Model to use: `perch_v2` or `perch_8` | `perch_v2` |
| `--output_path_type` | Preset output layout applied to both embeddings and recognizer results (overridden by more specific keys): `flat_basename`, `nested_basename`, `nested`, `flat` | None |
| `--dataset_name` | Dataset name used in runner configuration | `search_set` |
| `--db_path` | Database path; relative paths resolve under output | `db` |
| `--sourcemap_preset` | Optional preset used to rewrite the output `source` value (for embeddings and recognizer outputs) | None |
| `--sourcemap_token_vals` | Optional JSON object of token values injected into the selected sourcemap preset | None |
| `--save_db` | Persist the hoplite embedding database. Use --save_db with no value to enable (default: false) | `false` |
| `--file_glob` | Glob pattern for audio files, e.g. `*/*`, `*/*/*` | Auto-detected |
| `--workers` | Worker count or `auto` | `auto` |
| `--log_level` | App log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `--hoplite_log_level` | Library/root log level | `WARNING` |
| `--tf_log_level` | TensorFlow C++ log level | `WARNING` |
| `--log_file` | Optional log file path | None |
| `--config_file` | Path to a YAML config file | None |
| `--source` | Override source path (default: `/mnt/input`) | `/mnt/input` |
| `--output` | Override output path (default: `/mnt/output`) | `/mnt/output` |

### Analyze Option Details

#### --embed

Enables embedding export. This is a boolean flag.

- `--embed` or `--embed true`: enable embedding export.
- `--embed false`: explicitly disable (overrides a config file that enables it).
- Use `--embeddings_table_format` and `--embeddings_table_filetype` to control the output format.
- Setting any `--embeddings_*` key (e.g. `--embeddings_table_format columns`) implicitly enables embed unless `--embed false` is also set.

#### --embeddings_table_format

Controls table layout for embed outputs that do not explicitly include a table format.

- Allowed values: `serialized`, `columns`
- Multiple values allowed, comma-separated.

What they mean:

- `serialized`: one `embeddings` column containing a serialized vector per row.
- `columns`: one column per embedding dimension (`f0000`, `f0001`, ...).

#### --file_glob

Selects which audio files under source directory are embedded.

- If provided, it is used directly (examples: `*`, `*/*`, `*/*/*`).
- If omitted or false-like, it is auto-detected.
- recursive globbing is not possible, due to how perch-hoplite works
- This is ignored if the source points to a single file

Auto-detection behavior:

- The runner scans audio files recursively.
- It chooses a glob depth based on the shallowest audio file found:
  - top-level audio file -> `*`
  - one level deep -> `*/*`
  - two levels deep -> `*/*/*`
- Deeper files than the chosen depth are skipped.
- A warning is logged when deeper files are skipped.

Special case:

- If `--source` points to a single audio file, only that file is embedded (internally using the filename as the glob).

#### --embeddings_output_path_template

For the tabular (e.g. parquet) outputting of embeddings, you can specify where they are saved within `<output>/`
using a template.

- Supported tokens: 
  - `{parents}` the parent directories of the audio file, relative to the source directory
  - `{basename}` the basename of the audio file, without the extension
  - `{ext}` the extension of the output format, e.g. `.parquet`, `.csv`
  - `{embeddings_table_format}` the table format e.g. `serialized` or `columns`
  - `{analysis}` the output type — for embeddings this is always `embeddings`
- Must be a relative path.
- Must not contain `..` path traversal.

Examples:

- `{parents}/{basename}/{analysis}{ext}` (default — renders to e.g. `site1/recording.wav/embeddings.parquet`)
- `{parents}/{basename}/{embeddings_table_format}/embeddings{ext}`

If exporting both parquet table formats, include `{embeddings_table_format}` in the template to avoid path collisions.

If more than one source audio files map to the same output file, they will all be included in the same output file.

#### --embeddings_output_path_type

Preset output paths (mutually exclusive with `--embeddings_output_path_template`):

- `flat_basename` -> `{basename}{ext}`
- `nested_basename` -> `{parents}/{basename}{ext}`
- `nested` -> `{parents}/{basename}{ext}`
- `flat` -> `{analysis}{ext}` (all recordings in a single file, e.g. `embeddings.parquet`)

#### --recognizers

Runs embeddings through one or more linear classifiers (embeddings-classifier) and writes per-classifier result files.

- Accepts a path to a JSON file containing a `recognizers` list, or a bare list directly.
- Each recognizer produces its own output file per source recording by default.
- Output path controlled by `--recognizer_output_path_template` or `--recognizer_output_path_type`.
- When recognizers are configured, `model_choice` is derived automatically from the recognizer metadata unless explicitly set.

#### --recognizer_output_path_template

Output path template for recognizer result files. Controls where results are written within `<output>/`.

- Supported tokens:
  - `{recognizer_name}` the recognizer's name
  - `{parents}` parent directories of the source audio file
  - `{basename}` basename of the source audio file, without extension
  - `{ext}` output file extension, e.g. `.csv`, `.parquet`
  - `{analysis}` the output type — for recognizer results this is always `recognizer_results`
- Must be a relative path.
- Must not contain `..` path traversal.
- Default: `{recognizer_name}/{parents}/{basename}/{analysis}{ext}`

Examples:

- `{recognizer_name}/{parents}/{basename}/{analysis}{ext}` (default — one directory per recognizer, mirroring source structure)
- `{recognizer_name}/{analysis}{ext}` (one flat file per recognizer, all recordings merged)
- `{analysis}{ext}` (single file, all recognizers and recordings merged — use with care)

#### --recognizer_output_path_type

Preset output paths for recognizer results (mutually exclusive with `--recognizer_output_path_template`):

- `flat_basename` -> `{basename}{ext}`
- `nested_basename` -> `{parents}/{basename}{ext}`
- `nested` -> `{parents}/{basename}{ext}`
- `flat` -> `{analysis}{ext}` (all recordings for all recognizers in a single file)

Note: these presets do not include `{recognizer_name}`, so results from multiple recognizers will be merged into the same file. Add a `--recognizer_output_path_template` with `{recognizer_name}` if you need per-recognizer separation.

#### --output_path_type

Applies a preset layout to both embeddings and recognizer results at once. The more specific `embeddings_output_path_type` and `classify_output_path_type` take priority if also set.

- Accepted values: `flat_basename`, `nested_basename`, `nested`, `flat`
- Equivalent to setting both `--embeddings_output_path_type` and `--recognizer_output_path_type` to the same value.

#### --db_path

Location for the internal embedding database.

- Relative paths are resolved under `--output`.
- Default is `db`, which resolves to `<output>/db`.

#### --sourcemap_preset

Selects a hardcoded sourcemap preset that rewrites the exported `source` column.

- If unset, the original source path is written unchanged.
- Current preset: `canonical_name_to_original_recording_url`
- This preset extracts `audio_recording_id` from canonical names like `..._909057.flac` and renders:
  - `{domain}/audio_recordings/{audio_recording_id}/original`

Example:

- `analyze --embed --sourcemap_preset canonical_name_to_original_recording_url --sourcemap_token_vals '{"domain":"https://api.ecosounds.org"}'`

#### --sourcemap_token_vals

JSON object (CLI string or config file object) used to provide values for sourcemap template tokens.

- Requires `--sourcemap_preset`.
- Keys must be simple token names (letters, numbers, underscore).
- For `canonical_name_to_original_recording_url`, supply `domain`.

#### --save_db

Controls whether the hoplite embedding database is saved after processing.

- Behavior:
  - `--save_db true` (or `--save_db` with no value): Database is saved at the location specified by `--db_path`.
  - `--save_db false` (default): If the database folder specified by `--db_path` already exists, it is preserved and used. If the folder does not exist, a temporary database is created, used for exports, and then deleted.
- Validation:
  - At least one of `--embed`, `--classify`, or `--save_db` must be specified.
  - You can specify `--save_db true` without `--embed` to create and save only the database.

Usage examples:

- DB only: `analyze --save_db` (creates and saves the database, no embeddings exported)
- Embeddings only: `analyze --embed` (creates embeddings, database is deleted after)
- Both: `analyze --embed --save_db` (creates embeddings, saves database)

#### --workers

Controls embedding worker count passed to perch-hoplite

- `auto` (default): computed from available RAM.
- Integer value: explicit worker count. 

#### --classify [format]

Classification output selector.

- Allowed values: `parquet`, `csv`, `hoplite`
- `--classify` with no value defaults to `csv`
- Classification pipeline is currently not implemented.

#### --model_choice

Embedding model preset.

- Allowed values: `perch_v2`, `perch_8`

#### --dataset_name

Dataset name stored in runner configuration.

- Default: `search_set`
- This is currently a config value/CLI flag only; it does not change output path rendering unless some downstream consumer uses it.

#### --source and --output

Input and output roots.

- Both paths must already exist and mounted into the container.
- Source can be a directory or a single audio file.

#### --config_file

Path to config file (`.yml`, `.yaml`, or `.json`).

- CLI flags override values loaded from config file.

#### --log_level, --hoplite_log_level, --tf_log_level, --log_file

Logging controls.

- `--log_level`: perch-runner logs.
- `--hoplite_log_level`: library/root logs.
- `--tf_log_level`: TensorFlow C++ logs.
- `--log_file`: optional file output in addition to console.

### Supported Audio Formats

`.wav`, `.flac`, `.mp3`, `.ogg`

### Output Structure

Audio files are discovered relative to the source directory. Output mirrors that structure:

```
/mnt/output/
  site1/
    recording.wav/
      embeddings.parquet
  site2/
    another.flac/
      embeddings.parquet
```

Each Parquet file contains one row per 5-second window with columns: `source`, `channel`, `offset`, `embeddings` (serialized numpy array).

With `--embeddings_table_format columns`, the `embeddings` column is replaced by individual dimension columns (`f0000`, `f0001`, ...).

### Config File

Instead of CLI flags, you can mount a YAML config file:

```yaml
source: /mnt/input
output: /mnt/output
embed: parquet
model_choice: perch_v2
embeddings_table_format: serialized
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

or just paste:

```bash
IMAGE="${IMAGE:-qutecoacoustics/perchrunner:latest}"
docker run --rm --network=none --entrypoint /app/tests/run_tests "$IMAGE"
```


#### end-to-end tests from host

```bash
# run end-to-end tests only, on the host
./run_tests.sh
```

This runs host end-to-end tests in `tests/end_to_end_tests`, which execute the built
container via `docker run` and validate produced outputs on the host. 
Requires pytest and other python libraries to be installed on the host environment (see requirements-host.txt).  
It must be run from the root of the repo for test discovery and accessing the test files. 


## License

Apache 2.0

