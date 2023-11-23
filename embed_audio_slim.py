# embed a single audio file and save to a self-describing format
# we don't need to store information about the file that the example comes from

# Global imports
from pathlib import Path
import numpy as np
# import tqdm
import argparse
import pandas as pd
import soundfile
from math import ceil


from chirp import audio_utils

# we need to house config in this object because
# it's what TaxonomyModelTF expects. Specifically it needs an object that has dot 
# access to values as well as dict-like access for using the spread operator (**config) 
from ml_collections import config_dict

from chirp.inference.models import TaxonomyModelTF


parser = argparse.ArgumentParser()
parser.add_argument("--source_file", help="path to the file to analyze")
parser.add_argument("--output_folder", help="file to embeddings to")
parser.add_argument("--max_segments", default=-1, type=int, help="only analyse this many segments of the file. Useful for debugging quickly. If ommitted will analyse all")
parser.add_argument("--segment_length", default=60, type=int,  help="the file is split into segments of this duration in sections to save loading entire files into ram")
parser.add_argument("--hop_size", default=5, type=float,  help="create an 5 second embedding every this many seconds. Leave as default 5s for no overlap and no gaps.")

args = parser.parse_args()

# check audio exists and get the duration
audio_file = soundfile.SoundFile(args.source_file)
audio_duration = audio_file.frames / audio_file.samplerate
print(f'analysing {args.source_file} samples: {audio_file.frames}, sr: {audio_file.samplerate}, duration {audio_duration} sec')


# TODO: build into image
model_path = "/phil/bird-vocalization-classifier_3/"

model_config = config_dict.create(
    hop_size_s = args.hop_size,
    model_path = model_path,
    sample_rate = 32000,
    window_size_s = 5.0
)

output_folder = Path(args.output_folder)
output_folder.mkdir(exist_ok=True, parents=True)


print('\n\nLoading model(s)...')
#embedding_model = TaxonomyModelTF.from_config(config.embed_fn_config["model_config"])
embedding_model = TaxonomyModelTF.from_config(model_config)

# an empty array of the with zero rows to concatenate to
# 1280 embeddings plus one column for the offset_seconds
file_embeddings = np.empty((0,1,1281))

total_segments = ceil(audio_duration / args.segment_length)
num_segments = min(args.max_segments, total_segments) if args.max_segments > 1 else total_segments

for segment_num in range(num_segments):
  offset_s = args.segment_length * segment_num
  print(f'getting embeddings for offsets {offset_s} to {offset_s + args.segment_length}')

  audio = audio_utils.load_audio_window(
      args.source_file, offset_s, 
      model_config.sample_rate, args.segment_length
  )

  if len(audio) > 1:
    segment_embeddings = embedding_model.embed(audio)
    offsets = np.arange(0,60,5).reshape(12,1,1)
    segment_embeddings = np.concatenate((offsets + offset_s, segment_embeddings.embeddings), axis=2)
    file_embeddings = np.concatenate((file_embeddings, segment_embeddings), axis=0)
  else:
    print('no audio found')
  

for channel in range(file_embeddings.shape[1]):
    channel_embeddings = file_embeddings[:,channel,:]
    df = pd.DataFrame(channel_embeddings, columns=['offset'] + ['e' + str(i).zfill(3) for i in range(1280)])
    destination_filename = Path(args.output_folder) / Path(f"embeddings_{channel}.csv")


    df.to_csv(destination_filename, index=False)