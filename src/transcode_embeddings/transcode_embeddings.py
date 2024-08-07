# Takes embeddings from parquet files and writes them to TFRecord files

import tensorflow as tf
import pandas as pd
import numpy as np

from chirp.inference.tf_examples import EmbeddingsTFRecordMultiWriter, bytes_feature, int_feature, float_feature, serialize_tensor

from src.data_frames import df_to_embeddings

def get_parquet_file_list(parquet_folder):
  """
  Recursively finds all parquet files in a folder
  """
  return [f for f in parquet_folder.rglob('*.parquet')]

def transcode_from_parquet(parquet_filepaths, output_path):

  print(f"transcoding {len(parquet_filepaths)} parquet files to {output_path}")


  with EmbeddingsTFRecordMultiWriter(output_path, num_files=256) as writer:
    for i, fp in enumerate(parquet_filepaths):

      #print a dot without a newline every 10th file and 
      # print i of total every 100 files
      if i % 10 == 0:
        if i % 100 == 0:
          print(f"\n{i} of {len(parquet_filepaths)}")
        else:
          print('.', end='', flush=True)

      # read the parquet file with pandas
      embeddings_table = df_to_embeddings(pd.read_parquet(fp))
      embeddings = np.array(embeddings_table[:,:,2:1282], dtype=np.float16)
      #print(f"embeddings shape: {embeddings.shape}")
      embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float16)
      #print(f"embeddings shape: {embeddings.shape}")
      features = {
        'filename': bytes_feature(embeddings_table[0][0][0].encode()),
        'timestamp_s': float_feature(0.0),
        'embedding': bytes_feature(serialize_tensor(embeddings, tf.float16)),
        'embedding_shape': int_feature(tuple(embeddings.shape))
      }
      ex = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(ex.SerializeToString())

# def filename_to_url(filename, domain):

#   # filename is made of 3 parts: datetime, site, and file number, followed by a file extension
#   # the 3 parts are separated by underscores. The site name might also contain an underscore
#   # the datetime is in the format YYYYMMDDTHHmmssZ, file number is an integer, and the file extension is .parquet
#   # we need to contruct a url like this: https://[domain]/



#   return f"https://storage.googleapis.com/urban-sound-classification/{filename}"