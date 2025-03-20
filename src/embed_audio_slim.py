# embed a single audio file and save to a self-describing format
# we don't need to store information about the file that the example comes from

import argparse
import fnmatch
from math import ceil
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile

from chirp import audio_utils

# we need to house config in this object because
# it's what TaxonomyModelTF expects. Specifically it needs an object that has dot 
# access to values as well as dict-like access for using the spread operator (**config) 
from ml_collections import config_dict

from chirp.inference.models import TaxonomyModelTF

from src import data_frames
from src import baw_utils 

DTYPE_MAPPING = {
    16: np.float16,
    32: np.float32,
    64: np.float64
}

def merge_defaults(config: config_dict):
  """
  gets the config based on user-supplied config and if they are missing then uses defaults
  """

  merged_config = config_dict.create(
    hop_size = 5,
    segment_length = 60,
    max_segments = -1,
    bit_depth = 16,
  )

  if config is None:
    config = config_dict.create()

  for key in config:
    merged_config[key] = config[key]

  return merged_config


def get_audio_files(source_folder):
    """
    Find all audio files within a source folder with common audio extensions.
    Returns a list of Path objects relative to the source folder.
    
    Args:
        source_folder: Path-like object pointing to the directory to search
        
    Returns:
        List of Path objects relative to source_folder
    """
    audio_extensions = [
        'wav', 'mp3', 'flac', 'ogg', 'aiff', 'aif', 'au', 
        'caf', 'htk', 'svx', 'mat4', 'mat5', 'mpc', 'paf', 
        'pvf', 'rf64', 'sd2', 'sds', 'sf', 'voc', 'w64', 
        'wve', 'xi', 'raw', 'ircam', 'nist', 'wavex'
    ]
    
    source_folder = Path(source_folder)
    source_files = []
    
    for file in source_folder.rglob('*'):
        if file.is_file():
            for ext in audio_extensions:
                if fnmatch.fnmatch(file.name.lower(), f'*.{ext}'):
                    source_files.append(file.relative_to(source_folder))
                    break
    
    return source_files


def embed_folder(source_folder, output_folder, config: config_dict = None) -> None:
  """
  for each file in source_folder, embeds and then saves to output_folder with a name that matches the original 
  """

  source_files = get_audio_files(source_folder)

  if len(source_files) == 0:
    print(f'no audio files found in {source_folder}')

  for source_file in source_files:
    print(f'analysing {source_file}')
    embeddings = embed_one_file(Path(source_folder / source_file), config)
    dest = Path(output_folder / source_file).with_suffix('.parquet')
    save_embeddings(embeddings, dest, source_file)


def embed_file_and_save(source: str, destination: str, config: config_dict = None) -> None:
    """
    embeds a single file and saves to destination
    destination can be either a filename or a folder. If it's a folder that exists, the original basename is used with parquet extension
    """

    source = Path(source)
    destination = Path(destination)

    # check if destination is a directory which exists. 
    # If so, generate the filename to save as based on the source file basename,
    # defaulting to parquet as the output format
    if destination.is_dir():
       destination = destination / Path(source.name).with_suffix('.parquet')
    elif destination.suffix not in ('.parquet', '.csv'):
       raise ValueError(f"Invalid destination: {destination}. Must be a file with a valid extension or an existing directory")

    embeddings = embed_one_file(source, config)
    source = baw_utils.recording_url_from_filename(source)
    save_embeddings(embeddings, destination, source)


def parse_source(source: str, baw_host='api.ecosounds.org') -> str:
    """
    returns a string that is the source of the embeddings
    TODO (phil): implement this if needed
    """
    return str(source)


def embed_one_file(source: str, config: config_dict = None) -> np.array:

    config = merge_defaults(config)

    # check audio exists and get the duration
    audio_file = soundfile.SoundFile(source)
    audio_duration = audio_file.frames / audio_file.samplerate
    print(f'analysing {source} samples: {audio_file.frames}, sr: {audio_file.samplerate}, duration {audio_duration} sec')

    model_path = "/models/4"

    # model config contains some values from this function's config plus
    # some values we have fixed.
    model_config = config_dict.create(
        hop_size_s = config.hop_size,
        model_path = model_path,
        sample_rate = 32000,
        window_size_s = 5.0
    )

    print('\n\nLoading model(s)...')
    #embedding_model = TaxonomyModelTF.from_config(config.embed_fn_config["model_config"])
    embedding_model = TaxonomyModelTF.from_config(model_config)

    # an empty array of the with zero rows to concatenate to
    # 1280 embeddings plus one column for the offset_seconds
    # TODO: I think the shape will be different when we have audio separation channels
    file_embeddings = np.empty((0, 1, 1281), dtype=DTYPE_MAPPING[config.bit_depth])

    total_segments = ceil(audio_duration / config.segment_length)
    num_segments = min(config.max_segments, total_segments) if config.max_segments >= 1 else total_segments

    for segment_num in range(num_segments):
      offset_s = config.segment_length * segment_num
      print(f'getting embeddings for offsets {offset_s} to {offset_s + config.segment_length}')

      audio = audio_utils.load_audio_window(
          source, offset_s, 
          model_config.sample_rate, config.segment_length
      )

      if len(audio) > 1:
        embeddings = embedding_model.embed(audio).embeddings
        embeddings = embeddings.astype(DTYPE_MAPPING[config.bit_depth])
        # the last segment of the file might be smaller than the rest, so we always use its length
        # dtype should bee float32 to match the embeddings
        offsets = np.arange(0,embeddings.shape[0]*config.hop_size,config.hop_size, dtype=DTYPE_MAPPING[config.bit_depth]).reshape(embeddings.shape[0],1,1) + offset_s

        # if source separation was used, embeddings will have an extra dimention for channel
        # for consistency we will add the extra dimention even if there is only 1 channel

        shape = embeddings.shape
        if len(shape) == 2:
           embeddings = embeddings.reshape(shape[0], 1, shape[1])
        
        segment_embeddings = np.concatenate((offsets, embeddings), axis=2)
        file_embeddings = np.concatenate((file_embeddings, segment_embeddings), axis=0)
      else:
        print('no audio found')

    return file_embeddings


def save_embeddings(embeddings: np.array, destination: str, source: str=None, file_type=None):
    """
    saves the embeddings to the destination in a format
    @param embeddings: a numpy array of embeddings, of shape (num_segments, num_channels, 1281). The first column is the offset in seconds
    @param destination: the path to save the embeddings file to
    @param source: what to add in the source column. This is probably a workbench recording url or a path to the original file
    @param file_type: the type of file to save to. If None, will be determined from the extension of the destination
    """
    
    destination = Path(destination)
    print(f'creating output folder: {destination.parent}')
    destination.parent.mkdir(exist_ok=True, parents=True)
    embeddings_df = data_frames.embeddings_to_df(embeddings)

    if file_type is None:
       # determine from extension
       file_type = destination.suffix[1:]

    
    embeddings_df.insert(0, 'source', str(source))

    match file_type:
        case "parquet":
            print(f'saving as parquet to {destination}')
            embeddings_df.to_parquet(destination, index=False)
        case "csv":
            embeddings_df = data_frames.serialize_embeddings_df(embeddings_df)
            embeddings_df.to_csv(destination, index=False)
        case _:
            raise ValueError(f'Invalid file type for saving embeddings data frame: {file_type}')

