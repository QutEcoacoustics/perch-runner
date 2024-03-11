# perch-runner

A docker container that runs Perch.

## Commands

This container has three basic functions:

1. Generate Perch embeddings for a given audio file
2. To classify the embeddings given an audio file and one or more additional linear classifier
3. To measure distance between a given sample and a given audio file (Not implemented Yet)

All three commands output results in 5 second chunks.

At this time we're choosing not to reuse embeddings. It is easier to (though much much slower)
to ignore the caching and model version problems that come with reusing embeddings.

4. To train a linear classifier from a set of labelled audio clips, which will output a model file and json metadata file. 

### Generate embeddings

The `generate` command accepts one input file and one output directory.

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
generate [-F|--format] [-c|--config <config-file>] <input-file> <output-directory>
    <input-file>                The file to generate embeddings for
    <output-directory>          The directory to write the embeddings to
    -F|--format [csv|parquet]   The format of the embeddings file: `CSV`` or `Parquet``
    -c|--config <config-file>   An optional configuration file to use
```

#### Configuration file

```yaml
hop_length_seconds: 5.0
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
hop_length_seconds: 5.0
classifiers:
  - name: "classifier1"
    tags: ["tag1", "tag2"]
    # base64 encoded model
    model: "asd321jhfsdhk438asdmlfas89i3mnlksf..."
  - name: "classifier2"
    tags: ["tag3", "tag4"]
     # base64 encoded model
    model: "..."

```

#### Example

```bash 
cd /data
curl -o audio.wav https://api.ecosounds.org/audio_recordings/123.wav
docker run -v /data:/data perch-runner classify -c /data/config.yml /data/audio.wav /data/output
```

### Distance

The `distance` command accepts one input file, one output directory, and a query file.

- The query file must be an audio file that is 5 seconds long.
- The output of this command is the same as the output of the `generate` command,
    but with an additional column `distance` that represents the distance between the
    query and the sample.

#### Usage

```
distance [-F|--format] [-c|--config <config-file> ] <input-file> <query-file> <output-directory> 
    <input-file>                The file to generate embeddings for
    <output-directory>          The directory to write the embeddings to
    <query-file>                The file to use as the query
    -c|--config <config-file>   An optional configuration file to use
```

#### Configuration file

```yaml
hop_length_seconds: 5.0
```

#### Example

```bash
cd /data
curl -o audio.wav https://api.ecosounds.org/audio_recordings/123.wav
curl -o query.wav https://api.ecosounds.org/audio_recordings/456/media.wav?start=10&end=15
docker run -v /data:/data perch-runner distance -c /data/config.yml /data/audio.wav /data/query.wav /data/output
```


## Batch Mode

The "generate" and "inference" commands can be also be used in batch mode by altering the entrypoint to `/app/src/batch.py`


# Tests

## Host tests

To test the docker container works as expected from the host:
1. Into a virtual environment install the testing dependencies with `pip install -r requirements-host.txt`
2. Run `pytest 

