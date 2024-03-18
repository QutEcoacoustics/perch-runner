# from the last part of https://github.com/google-research/perch/blob/main/active_learning.ipynb


#@title Imports. { vertical-output: true }
import collections
import json
from ml_collections import config_dict
import numpy as np
from etils import epath
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import keras

from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)
#
from chirp.inference import tf_examples
from chirp.models import metrics
from chirp.projects.bootstrap import bootstrap
from chirp.projects.bootstrap import search
from chirp.projects.bootstrap import display
from chirp.projects.multicluster import classify
from chirp.projects.multicluster import data_lib

#@title Configure data and model locations. { vertical-output: true }

# Path containing TFRecords of unlabeled embeddings.
# We will load the model which was used to compute the embeddings automatically.
embeddings_path = '/phil/output/pw_embeddings_all/' #@param



model_path = "/output/trained_model.keras"


model = keras.models.load_model(model_path)



# #@title Load the model. { vertical-output: true }

# Get relevant info from the embedding configuration.
embeddings_path = epath.Path(embeddings_path)
with (embeddings_path / 'config.json').open() as f:
  embedding_config = config_dict.ConfigDict(json.loads(f.read()))
embeddings_glob = embeddings_path / 'embeddings-*'

config = bootstrap.BootstrapConfig.load_from_embedding_config(
    embeddings_path=embeddings_path,
    # we don't need annotated data for this but the bootstrap config requires it
    annotated_path='')
embedding_hop_size_s = config.embedding_hop_size_s
project_state = bootstrap.BootstrapState(config)
# do we need this? we shouldn't
embedding_model = project_state.embedding_model

print(embedding_model)




# #@title Run model on target unlabeled data. { vertical-output: true }

# Choose the target class to work with.
# TODO: we shouldn't need this, I think it's just for the active learning part
target_class = 'pos'  #@param
# Choose a target logit; will display results close to the target.
# Set to None to get the highest-logit examples.
target_logit = None  #@param
# Number of results to display.
# we actually want to save ALL results, in fact we want to save All logits, right?
num_results = 25  #@param

# # Create the embeddings dataset.
embeddings_ds = tf_examples.create_embeddings_dataset(
    embeddings_path, file_glob='embeddings-*')

# in the active learning, this is taken from the merged dataset from folder_of_folders
# we need to be able to find this out
# target_class_idx = merged.labels.index(target_class)
target_class_idx = 0

results, all_logits = search.classifer_search_embeddings_parallel(
    embeddings_classifier=model,
    target_index=target_class_idx,
    embeddings_dataset=embeddings_ds,
    hop_size_s=embedding_hop_size_s,
    target_score=target_logit,
    top_k=num_results
)

# results is a results object from the chirp bootstrap
print(results)

results.write_labeled_data(config.annotated_path, embedding_model.sample_rat)

# this is just a 1d np array. it's not very useful alone because we can't match the
# index of this array to a particular file-offset without more info
print(all_logits)


# def plot_logits(target_class, all_logits, bins=128):
  
#     # Plot the histogram of logits.
#     _, ys, _ = plt.hist(all_logits, bins=bins, density=True)
#     plt.xlabel(f'{target_class} logit')
#     plt.ylabel('density')
#     # plt.yscale('log')
#     plt.plot([target_logit, target_logit], [0.0, np.max(ys)], 'r:')
#     plt.show()



# #@title Display results for the target label. { vertical-output: true }

# display_labels = merged.labels

# extra_labels = []  #@param
# for label in extra_labels:
#   if label not in merged.labels:
#     display_labels += (label,)
# if 'unknown' not in merged.labels:
#   display_labels += ('unknown',)

# display.display_search_results(
#     results, embedding_model.sample_rate,
#     project_state.source_map,
#     checkbox_labels=display_labels,
#     max_workers=5)


# #@title Add selected results to the labeled data. { vertical-output: true }

# results.write_labeled_data(
#     config.annotated_path, embedding_model.sample_rate)


# #@title Write classifier inference CSV. { vertical-output: true }

# threshold = 1.0  #@param
# output_filepath = '/tmp/inference.csv'  #@param

# # Create the embeddings dataset.
# embeddings_ds = tf_examples.create_embeddings_dataset(
#     embeddings_path, file_glob='embeddings-*')

# def classify_batch(batch):
#   """Classify a batch of embeddings."""
#   emb = batch[tf_examples.EMBEDDING]
#   emb_shape = tf.shape(emb)
#   flat_emb = tf.reshape(emb, [-1, emb_shape[-1]])
#   logits = model(flat_emb)
#   logits = tf.reshape(
#       logits, [emb_shape[0], emb_shape[1], tf.shape(logits)[-1]])
#   # Take the maximum logit over channels.
#   logits = tf.reduce_max(logits, axis=-2)
#   batch['logits'] = logits
#   return batch

# inference_ds = tf_examples.create_embeddings_dataset(
#     embeddings_path, file_glob='embeddings-*')
# inference_ds = inference_ds.map(
#     classify_batch, num_parallel_calls=tf.data.AUTOTUNE
# )

# with open(output_filepath, 'w') as f:
#   # Write column headers.
#   headers = ['filename', 'timestamp_s', 'label', 'logit']
#   f.write(', '.join(headers) + '\n')
#   for ex in tqdm.tqdm(inference_ds.as_numpy_iterator()):
#     for t in range(ex['logits'].shape[0]):
#       for i, label in enumerate(merged.class_names):
#         if ex['logits'][t, i] > threshold:
#           offset = ex['timestamp_s'] + t * config.embedding_hop_size_s
#           logit = '{:.2f}'.format(ex['logits'][t, i])
#           row = [ex['filename'].decode('utf-8'),
#                  '{:.2f}'.format(offset),
#                  label, logit]
#           f.write(', '.join(row) + '\n')