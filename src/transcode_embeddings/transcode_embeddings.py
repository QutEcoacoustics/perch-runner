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

def transcode_from_parquet(parquet_filepaths, output_path, num_files=10):

  print(f"transcoding {len(parquet_filepaths)} parquet files to {output_path}")


  with EmbeddingsTFRecordMultiWriter(output_path, num_files=num_files) as writer:
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
      embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float16)
      features = {
        'filename': bytes_feature(embeddings_table[0][0][0].encode()),
        'timestamp_s': float_feature(0.0),
        'embedding': bytes_feature(serialize_tensor(embeddings, tf.float16)),
        'embedding_shape': int_feature(tuple(embeddings.shape))
      }
      ex = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(ex.SerializeToString())
