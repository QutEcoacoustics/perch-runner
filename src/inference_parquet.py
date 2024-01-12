# in this file, we attempt to run inference with as few deps as possible
# we we will drill into the classes and functions invoked by the active_learning notebook.
# it may therefore have more code, as more code will need to be copy-pasted


import keras
import json
from pathlib import Path
import pandas as pd

# progress bar
import tqdm

import tensorflow as tf

# utilities for mapping tf_examples. 
# this is probably needed because it is used during the embedding stage
# and for things like referring to keys in the embedding metadata we 
# use constants defined here
from chirp.inference import tf_examples
from data_frames import df_to_embeddings

model_path = "/output/trained_model.keras"
embeddings_classifier = keras.models.load_model(model_path)



def read_parquet_files(folder_path):
    # Create a Path object for the folder
    folder = Path(folder_path)
    
    # List to hold dataframes
    dfs = []
    
    # Loop through all parquet files in the folder
    for file_path in folder.rglob('*.parquet'):
        df = pd.read_parquet(file_path)
        dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

   
def create_tf_dataset(dataframe, rows_per_item=12):
    """
    @param rows_per_item. The number of rows in the dataframe that correspond to a single dataset item
    """

    dataset_elements = []
    
    # convert to a dict of dataframes, one for each source (and remove the source column from the dataframe)
    source_files = {name: group.drop('source', axis=1) for name, group in dataframe.groupby('source')}


    for source, df in source_files.items():
       
        embeddings = df_to_embeddings(df)

        # for a given channel, offsets will be the same, so we can just take the first channel
        offsets = embeddings[:, 0, 0]
        # all rows (dim1) all channels (dim2), all features (dim3 without the offset)
        features = embeddings[:, :, 1:]

        for start_row in range(0, len(features), 12):
            end_row = min(start_row + 12, features.shape[0])
            item = {
                'source': source,
                'offsets': offsets[start_row:end_row],
                'embeddings': features
            }
            dataset_elements.append(item)

    dataset = tf.data.Dataset.from_generator(
      lambda: iter(dataset_elements),
      output_types={'source': tf.string, 'offsets': tf.float32, 'embeddings': tf.float32}
      )

       

    



    feature_columns = [f'f{i:04d}' for i in range(1280)]
    metadata_columns = ['source', 'offset', 'channel']
    features_df = dataframe[feature_columns].values
    metadata = dataframe[metadata_columns]

    features_tensor = tf.convert_to_tensor(features_df, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({"features": features_tensor, 
                                                   "metadata": metadata.to_dict('list')}))
    return dataset

def create_tf_dataset_old(dataframe, rows_per_item=12):
    """
    @param rows_per_item. The number of rows in the dataframe that correspond to a single dataset item
    """
    
    # convert to a dict of dataframes, one for each source
    source_files = {name: group for name, group in dataframe.groupby('source')}
    
    feature_columns = [f'f{i:04d}' for i in range(1280)]
    metadata_columns = ['source', 'offset', 'channel']
    features_df = dataframe[feature_columns].values
    metadata = dataframe[metadata_columns]

    features_tensor = tf.convert_to_tensor(features_df, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({"features": features_tensor, 
                                                   "metadata": metadata.to_dict('list')}))
    return dataset


#@title Configure data and model locations. { vertical-output: true }




# path to folder of parquet files
embeddings_path_2 = '/phil/output/cgw/file_embeddings/cgw_embeddings/20230526/'

combined_embeddings_df = read_parquet_files(embeddings_path_2)
embeddings_dataset = create_tf_dataset(combined_embeddings_df)


# the embeddings path contains the embeddings files themselves plus a 
# json file with some necessary?? metadata
embeddings_path = '/phil/output/pw_embeddings_all/'
embeddings_path = Path(embeddings_path)
with (embeddings_path / 'config.json').open() as embeddings_config_json:
  embeddings_config = json.loads(embeddings_config_json.read())

print(embeddings_config)

embeddings_ds = tf_examples.create_embeddings_dataset(
    embeddings_path, file_glob='embeddings-*')


def classify_batch(batch):
    """
    2nd map function
    """
    emb = batch[tf_examples.EMBEDDING]
    emb_shape = tf.shape(emb)
    flat_emb = tf.reshape(emb, [-1, emb_shape[-1]])
    # this is where we actually run the forward pass
    logits = embeddings_classifier(flat_emb)
    logits = tf.reshape(
        logits, [emb_shape[0], emb_shape[1], tf.shape(logits)[-1]]
    )
    # Restrict to target class.
    # logits = logits[..., target_index]
    # Take the maximum logit over channels.
    logits = tf.reduce_max(logits, axis=-1)
    batch['scores'] = logits
    return batch

# I don't think this actually processes anything yet, just 
# adds the map function to the pipeline. docs say something about 
# eager vs lazy which might have something to do with it
embeddings_ds = embeddings_ds.map(classify_batch)


all_distances = []
all_results = []


try:
    # iterate over the examples, which are the embedding files
    for ex in tqdm.tqdm(embeddings_ds.as_numpy_iterator()):
      all_distances.append(ex['scores'].reshape([-1]))
      
      # iterate over the segments within the embedding file
      for t in range(ex[tf_examples.EMBEDDING].shape[0]):
        offset_s = t * embeddings_config["embed_fn_config"]["model_config"]["hop_size_s"] + ex[tf_examples.TIMESTAMP_S]
 
        result = {
           "embedding": ex[tf_examples.EMBEDDING][t, :, :],
           "score": ex['scores'][t],
           "filename": ex['filename'].decode(),
           "offset_seconds": offset_s
        }
        
        print(f'{result["filename"]} {result["offset_seconds"]}')
        # print('.', end='')

        all_results.append(result)

      # print('+', end='')
   
except KeyboardInterrupt:
    pass


