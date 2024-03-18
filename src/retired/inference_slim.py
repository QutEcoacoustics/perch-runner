# in this file, we attempt to run inference with as few deps as possible
# we we will drill into the classes and functions invoked by the active_learning notebook.
# it may therefore have more code, as more code will need to be copy-pasted


import keras
import json
from pathlib import Path

# progress bar
import tqdm

import tensorflow as tf

# utilities for mapping tf_examples. 
# this is probably needed because it is used during the embedding stage
# and for things like referring to keys in the embedding metadata we 
# use constants defined here
from chirp.inference import tf_examples

model_path = "/output/trained_model.keras"
embeddings_classifier = keras.models.load_model(model_path)


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


