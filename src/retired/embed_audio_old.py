## duplication of the embed_audio.ipynb notebook, so it's easier to run with breakpoints in vscode

#@title Imports. { vertical-output: true }

# Global imports
from etils import epath
import numpy as np
import tensorflow as tf
import tqdm
from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)
import argparse

from chirp import audio_utils
from chirp import config_utils
from chirp.configs import config_globals
from chirp.inference import embed_lib
from chirp.inference import tf_examples


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")

args = parser.parse_args()


# Name of base configuration file in `chirp/inference/configs`
config_key = 'raw_soundscapes'  #@param
config = embed_lib.get_config(config_key)
config = config_utils.parse_config(config, config_globals.get_globals())

# Here we adjust the input and output targets.
# config.output_dir = 'output/all_02'  #@param
# config.source_file_patterns = ['/napco_survey_project_audio/PW/inference_datasets/20230413/*/*/*.wav']  #@param

config.output_dir = args.output  #@param
config.source_file_patterns = [args.input]  #@param



# Define the model
model_choice = 'perch'
# For Perch, the directory containing the model.
# For BirdNET, point to the specific tflite file.
model_path = "/phil/bird-vocalization-classifier_3/"  #@param
config.embed_fn_config.model_config.model_path = model_path
if model_choice == 'perch':
  config.embed_fn_config.model_config.window_size_s = 5.0
  config.embed_fn_config.model_config.hop_size_s = 5.0
  config.embed_fn_config.model_config.sample_rate = 32000
elif model_choice == 'birdnet':
  config.embed_fn_config.model_config.window_size_s = 3.0
  config.embed_fn_config.model_config.hop_size_s = 3.0
  config.embed_fn_config.model_config.sample_rate = 16000

# Only write embeddings to reduce size.
config.embed_fn_config.write_embeddings = True
config.embed_fn_config.write_logits = False
config.embed_fn_config.write_separated_audio = False
config.embed_fn_config.write_raw_audio = False


# Embedding windows are broken up into groups, typically one minute in length.
# This lets us limit input size to the model, track progres and
# recover from failures more easily.
config.shard_len_s = 60  #@param
config.num_shards_per_file = 1  #@param

# Number of parent directories to include in the filename.
config.embed_fn_config.file_id_depth = 2

# Number of TF Record files to create.
config.tf_record_shards = 10  #@param

# Speech filter threshold for YamNet.
# Set to a value between 0 and 1, or -1 to disable.
config.embed_fn_config.speech_filter_threshold = -1.0

#@title Set up. { vertical-output: true }

# Create output directory and write the configuration.
output_dir = epath.Path(config.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
embed_lib.maybe_write_config(config, output_dir)

# Create SourceInfos.
source_infos = embed_lib.create_source_infos(
    config.source_file_patterns,
    config.num_shards_per_file,
    config.shard_len_s)
print(f'Found {len(source_infos)} source infos.')

# Set up the embedding function, including loading models.
embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
print('\n\nLoading model(s)...')
embed_fn.setup()

print('\n\nTest-run of model...')
# We run the test twice - the first run optimizes the execution, and
# subsequent runs will be full-speed.
window_size_s = config.embed_fn_config.model_config.window_size_s
sr = config.embed_fn_config.model_config.sample_rate
z = np.zeros([int(sr * window_size_s)])
embed_fn.embedding_model.embed(z)
print('Setup complete!')

#@title Run embedding. (safe) { vertical-output: true }

# Loads audio files one-by-one using methods which will tend not to fail
# if the target files have minor problems (eg, wrong length metadata).

embed_fn.min_audio_s = 1.0
record_file = (output_dir / 'embeddings.tfrecord').as_posix()
succ, fail = 0, 0
with tf_examples.EmbeddingsTFRecordMultiWriter(
    output_dir=output_dir, num_files=config.tf_record_shards) as file_writer:
  for source_info in tqdm.tqdm(source_infos):
    examples = embed_fn.process(source_info=source_info)
    if examples is None:
      fail += 1
      continue
    for example in examples:
      file_writer.write(example.SerializeToString())
    succ += 1
  file_writer.flush()
print(f'\n\nSuccessfully processed {succ} source_infos, failed {fail} times.')

fns = [fn for fn in output_dir.glob('embeddings-*')]
ds = tf.data.TFRecordDataset(fns)
parser = tf_examples.get_example_parser()
ds = ds.map(parser)
for ex in ds.as_numpy_iterator():
  print(ex['filename'])
  print(ex['embedding'].shape)
  break

#@title Run embedding. (fast) { vertical-output: true }

# Uses multiple threads to load audio before embedding.
# This tends to be faster, but can fail if any audio files are corrupt.

embed_fn.min_audio_s = 1.0
record_file = (output_dir / 'embeddings.tfrecord').as_posix()
succ, fail = 0, 0

audio_iterator = audio_utils.multi_load_audio_window(
    filepaths=[s.filepath for s in source_infos],
    offsets=[s.shard_num * s.shard_len_s for s in source_infos],
    sample_rate=config.embed_fn_config.model_config.sample_rate,
    window_size_s=config.shard_len_s,
)
with tf_examples.EmbeddingsTFRecordMultiWriter(
    output_dir=output_dir, num_files=config.tf_record_shards) as file_writer:
  for source_info, audio in tqdm.tqdm(
      zip(source_infos, audio_iterator), total=len(source_infos)):
    file_id = source_info.file_id(config.embed_fn_config.file_id_depth)
    offset_s = source_info.shard_num * source_info.shard_len_s
    example = embed_fn.audio_to_example(file_id, offset_s, audio)
    if example is None:
      fail += 1
      continue
    file_writer.write(example.SerializeToString())
    succ += 1
  file_writer.flush()
print(f'\n\nSuccessfully processed {succ} source_infos, failed {fail} times.')

fns = [fn for fn in output_dir.glob('embeddings-*')]
ds = tf.data.TFRecordDataset(fns)
parser = tf_examples.get_example_parser()
ds = ds.map(parser)
for ex in ds.as_numpy_iterator():
  print(ex['filename'])
  print(ex['embedding'].shape)
  break