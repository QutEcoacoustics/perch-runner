# perch-runner

A docker container that runs Perch.

## Commands

This container has three basic functions:

1. Generate Perch embeddings for a given audio file
2. To classify the embeddings given an audio file and one or more additional linear classifier
3. To measure distance between a given sample and a given audio file (Not implemented Yet)

Each of these output a result per 5 second chunks

### Generate embeddings

The `generate` command accepts one input file or one input folder and one output directory.

- One file will be produced containing the embeddings for the input file.
- The file MUST be in a self-describing format. 
- Each result in the file must have at least
  - a `start` value representing the start of the record in seconds
  - an `embedding` value representing the embedding for the record
    - For CSV this should be a Base64 encoded string of the embedding vector
  - a `source_separation_channel` representing the virtual channel the record came from
  - and other columns are allowed as needed
- Files should include provenance metadata in their headers
- Note: audio recordings IDs (or other source identifying information) are
  not needed. The assumption is that this container is run by an orchestrator
  that knows what audio file is being processed.

#### Usage

```
generate <input-file> <output-directory>
    <input-file>                The file to generate embeddings for
    <output-directory>          The directory to write the embeddings to
    -F|--format [csv|parquet]   The format of the embeddings file: `CSV`` or `Parquet``
    -c|--config <config-file>   An optional configuration file to use
```



#### Examples

```bash
cd /data
curl -o audio.wav https://api.ecosounds.org/audio_recordings/123.wav
docker run -v /data:/data perch-runner generate /data/audio.wav /data/output
```

### Classify

The `classify` command accepts one input file, one output directory, and one or more classifiers.
The classifiers are provided in a config file.

- One file will be produced containing the classifications for the input file.
- The file MUST be in a self-describing format.
- The file should return results in \<insert link to our event common format\>
- Files should include provenance metadata in their headers

#### Usage

```
classify [-F|--format] [-c|--config <config-file> ] <input-file> <output-directory>
    <input-file>                The file to generate embeddings for
    <output-directory>          The directory to write the embeddings to
    -c|--config <config-file>   An optional configuration file to use
```

#### Configuration file

```yaml
classifier: keras_saved_model_path
```



## Batch Mode

The "generate" and "inference" commands can be also be used in batch mode by altering the entrypoint to `/app/src/batch.py`


# Tests

## Host tests

To test the docker container works as expected from the host:
1. Into a virtual environment install the testing dependencies with `pip install -r requirements-host.txt`
2. Run `pytest 

