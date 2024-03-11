# run inference file by file on parquet embeddings files
# each input parquet file will result in a csv file with the same name

import argparse
import json
from pathlib import Path
from functools import partial
import random

import keras
from ml_collections import config_dict
import pandas as pd
import tensorflow as tf
# progress bar
import tqdm

# utilities for mapping tf_examples. 
# this is probably needed because it is used during the embedding stage
# and for things like referring to keys in the embedding metadata we 
# use constants defined here
from chirp.inference import tf_examples
from data_frames import df_to_embeddings


def load_model(model_path):
    model = keras.models.load_model(model_path)
    with open(model_path + '.labels.json') as f:
        labels = tuple(json.load(f))
    return model, labels



def process_embeddings(embeddings_path, classifier_model, output_path, skip_if_file_exists=False):
    """
    @param classifier_model: either a tuple of (model, labels) or a path to a saved model 
                             (where saved model labels path is model_path + '.labels.json' by convention). 
    """
    if isinstance(classifier_model, str):
       embeddings_classifier, labels = load_model(classifier_model)
    else:
       embeddings_classifier, labels = classifier_model

    # list of paths to the embeddings files relative to the embeddings_path
    embeddings_files_relative = [path.relative_to(embeddings_path) for path in Path(embeddings_path).rglob('*.parquet')]

    # poor-person's parallel: shuffle and start script in a different process with skip_if_file_exists=True
    random.shuffle(embeddings_files_relative)

    print(f'processing {len(embeddings_files_relative)} embeddings files')

    for index, embedding_file in enumerate(tqdm.tqdm(embeddings_files_relative, desc="Processing")):

        file_output_path = output_path / embedding_file.with_suffix('.csv')
        if skip_if_file_exists and file_output_path.exists():
            #print(f'skipping {embedding_file} as {file_output_path} already exists')
            continue
        #print(f'processing {index} of {len(embeddings_files_relative)}: {embedding_file}')
        df = pd.read_parquet(embeddings_path / embedding_file)
        embeddings_ds = create_tf_dataset(df)
        #debug_classifier(embeddings_ds, embeddings_classifier)
        # lazy map doesn't get executed until we iterate over the dataset
        classify_function = partial(classify_batch, embeddings_classifier=embeddings_classifier)
        embeddings_ds = embeddings_ds.map(classify_function)
        # these must match the order of the classes in the classifier model
        # TODO: save these with and retrieve from the model

        results = classify_items(embeddings_ds, labels)
        # just get the scores, fname and offset. no distances (not sure what this even is) or embeddings
        results_df = pd.DataFrame(results, columns=['filename', 'offset_seconds'] + list(labels))

        file_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(file_output_path, index=False)

    print(f'finished processing {len(embeddings_files_relative)} embeddings files')
    


def debug_classifier(ds, model):
    """
    just a little function to test the classifier on a single batch
    explicitly instead of by adding a map function to the dataset, so it's easier to 
    debug 
    """

    model_2cl = keras.models.load_model('/phil/output/trained_model.keras')
    model_3cl = keras.models.load_model('/phil/output/trained_model_3cl.keras')

    exes = list(ds.take(2))

    emb_single = exes[0][tf_examples.EMBEDDING]
    emb_double = tf.concat([emb_single, exes[1][tf_examples.EMBEDDING]], axis=1)

    ex_single = {tf_examples.EMBEDDING: emb_single}
    emb_double = {tf_examples.EMBEDDING: emb_double}

    # classify a single example both 1 and 2 channels with both 2 and 3 class models
    classify_batch(exes[0], model, 1)
    classify_batch(ex_single, model_2cl, 1)
    classify_batch(ex_single, model_3cl, 1)
    classify_batch(emb_double, model_2cl, 1)
    classify_batch(emb_double, model_3cl, 1)

    # emb2_shape = [12, 2, 1280]
    # emb2 = tf.reshape(tf.range(emb2_shape[0] * emb2_shape[1] * emb2_shape[2]), emb2_shape)
    # flat_emb2 = tf.reshape(emb2, [-1, emb2_shape[-1]]) 


   

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
    Create a tf.data.Dataset from a dataframe of embeddings
    Assumes that for a given source, time offsets are sorted and and for a given source, time_offset, channels are sorted
    @param rows_per_item. The number of rows in the dataframe that correspond to a single dataset item
                          The dataset is created like this to match closer to the orignal active learning notebook. It does
                          seem to complicate things somewhat though. 
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

        for start_row in range(0, len(features), rows_per_item):
            # account for the fact that the last item may not have the full number of rows
            end_row = min(start_row + rows_per_item, features.shape[0])
            item_embedding = features[start_row:end_row, :, :]
            item = {
                'filename': source,
                'timestamp_s': offsets[start_row:end_row],
                'embedding': features[start_row:end_row, :, :],
                'embedding_shape': tf.constant(item_embedding.shape, dtype=tf.int64),
                'raw_audio': tf.constant([], dtype=tf.float32),
                'raw_audio_shape': tf.constant([], dtype=tf.int64),
                'separated_audio': tf.constant([], dtype=tf.float32),
                'separated_audio_shape': tf.constant([], dtype=tf.int64)
            }
            dataset_elements.append(item)

    dataset = tf.data.Dataset.from_generator(
      lambda: iter(dataset_elements),
      output_types={'filename': tf.string, 'timestamp_s': tf.float32, 'embedding': tf.float32, 
                    'embedding_shape': tf.int64, 'raw_audio': tf.float32, 'raw_audio_shape': tf.int64, 
                    'separated_audio': tf.float32, 'separated_audio_shape': tf.int64},
                    # not sure about timestamp_s type. it's a scalar according to the element_spec of the original dataset from tf_records
                    # (and was always 0.0) but I feel like it needs to be an array of dim 1 and length
      # not sure if this is necessary. I did it to match to the original dataset from tf_records. 
      # Those cases where it is None I think is due to the item having unknown dimension due to the final item not having the full number of rows
      output_shapes={'filename': (), 'timestamp_s': None, 'embedding': None, 
                          'embedding_shape': (None,), 'raw_audio': None,  'raw_audio_shape': (None,), 
                          'separated_audio': None, 'separated_audio_shape': (None,)}
    )

    return dataset


def classify_batch(batch, embeddings_classifier):
    """
    based on the original classify_batch function in perch search
    except we don't have a target_index, we just return scores for all classes
    """
    emb = batch[tf_examples.EMBEDDING]
    emb_shape = tf.shape(emb)
    # flat_emb: 2d array of shape (num_segments * num_channels, num_features)
    # i.e. we stack the channels one under the other
    flat_emb = tf.reshape(emb, [-1, emb_shape[-1]]) 
    # this is where we actually run the forward pass
    logits = embeddings_classifier(flat_emb) # 2d array of shape [num_segmends * num_channels, num_classes]
    # this reshapes back into a 3d array of shape [num_segments, num_channels, num_classes]
    logits = tf.reshape(
        logits, [emb_shape[0], emb_shape[1], tf.shape(logits)[-1]]
    ) 
    # Restrict to target class by selecting only the target class index from the last dimension.
    # logits = logits[..., target_index]
    # Take the maximum logit over channels, which is dimension 1
    logits = tf.reduce_max(logits, axis=1)
    batch['scores'] = logits
    return batch

 
def classify_items(embeddings_ds, label_names, use_progress_bar = False):

    #all_distances = []
    all_results = []

    try:

        # we are already using a progress bar for the loop over files, so probably don't use one here, 
        # but it's an option if we want it
        wrapped_iterable = tqdm.tqdm(embeddings_ds.as_numpy_iterator()) if use_progress_bar else embeddings_ds.as_numpy_iterator()

        # iterate over the examples, which are the embedding files
        for ex in wrapped_iterable:
            #distances = ex['scores'].reshape([-1])
            #all_distances.extend(distances)

            # iterate over the segments within the embedding file
            # for t in range(ex[tf_examples.EMBEDDING].shape[0]):
            #     offset_s = t * embeddings_config["embed_fn_config"]["model_config"]["hop_size_s"] + ex[tf_examples.TIMESTAMP_S]
            for t in range(len(ex["timestamp_s"])):
                offset_s = ex["timestamp_s"][t]
                result = {
                    "embedding": ex[tf_examples.EMBEDDING][t, :, :],
                    "filename": ex['filename'].decode(),
                    "offset_seconds": offset_s
                }

                for i, label in enumerate(label_names):
                    result[label] = ex['scores'][t, i]
                  
                #print(f'{result["filename"]} {result["offset_seconds"]}')
   
                all_results.append(result)
            
            

    
    except KeyboardInterrupt:
        pass

    return all_results


def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", help="path to directory of embeddings files")
    parser.add_argument("--model_path", help="path to the saved classifier model")
    parser.add_argument("--output_dir", help="save the results here")
    parser.add_argument("--skip_if_file_exists", action='store_true', help="skip processing if the output file already exists")
    args = parser.parse_args()
    process_embeddings(args.embeddings_dir, args.model_path, args.output_dir, args.skip_if_file_exists)


if __name__ == "__main__":
    main()
