#@title Imports. { vertical-output: true }
import collections
import json
from ml_collections import config_dict
import numpy as np
from etils import epath
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

from pathlib import Path
import pandas as pd

#
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

def create_tf_dataset(dataframe):
    
    feature_columns = [f'f{i:04d}' for i in range(1280)]
    metadata_columns = ['source', 'offset', 'channel']
    features_df = dataframe[feature_columns].values
    metadata = dataframe[metadata_columns]

    features_tensor = tf.convert_to_tensor(features_df, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({"features": features_tensor, 
                                                   "metadata": metadata.to_dict('list')}))
    return dataset


#@title Configure data and model locations. { vertical-output: true }


embeddings_path = '/phil/output/pw_embeddings_all/'  #@param

# path to folder of parquet files
embeddings_path_2 = '/phil/output/cgw/file_embeddings/cgw_embeddings/20230526/'

combined_embeddings_df = read_parquet_files(embeddings_path_2)
embeddings_dataset = create_tf_dataset(combined_embeddings_df)


# Path to the labeled wav data.
# Should be in 'folder-of-folders' format - a folder with sub-folders for
# each class of interest.
# Audio in sub-folders should be wav files.
# Audio should ideally be 5s audio clips, but the system is quite forgiving.
labeled_data_path = '/phil/cgw_labelled/all_merged/'  #@param
labeled_data_path = '/phil/cgw_labelled/tiny_subset/'  #@param

#@title Load the model. { vertical-output: true }

# Get relevant info from the embedding configuration.
embeddings_path = epath.Path(embeddings_path)
with (embeddings_path / 'config.json').open() as f:
  embedding_config = config_dict.ConfigDict(json.loads(f.read()))
embeddings_glob = embeddings_path / 'embeddings-*'
# this is just a dataclass which is basically the embedding config but reshaped a bit, and the annotated path added
config = bootstrap.BootstrapConfig.load_from_embedding_config(
    embeddings_path=embeddings_path,
    annotated_path=labeled_data_path)
embedding_hop_size_s = config.embedding_hop_size_s


# this is a dataclass with
# - the config
# - the model
# - the embeddings dataset of unlabelled audio. 
# - the 'source infos' which has an entry per 'shard' i.e. per 60s of audio. 

project_state = bootstrap.BootstrapState(config)
embedding_model = project_state.embedding_model


dataset = project_state.embeddings_dataset
# the dataset items have the structure as follows (we might need to emulate this)
# 'embedding', numpy.ndarray of shape (12,1,1280) (note: not a tensor)
# 'embedding_shape', array with values [12,1,1280]
# 'filename', byte string e.g. b'032/20220917_AAO_-23.54115+140.19473/20220917T183300+1000_SunsetToSunrise_-23.54115+140.19473.wav'
# 'raw_audio', numpy array float32. in this case it's empty.
# 'raw_audio_shape', numpy array float32. in this case it's empty.
# 'separated_audio', numpy array float32. in this case it's empty. 
# 'separated_audio_shape', tensor shape (0,) (when I try to inspec this the debug server disconnects)
# 'timestamp_s' tensor of shape (). All seems to have value 0.0, but this might be cause of a mistake I made during embedding to only do the 1st min?
# items = list(dataset.take(1).as_numpy_iterator())

# @title Load+Embed the Labeled Dataset. { vertical-output: true }

# Time-pooling strategy for examples longer than the model's window size.
time_pooling = 'mean'  # @param

# this is the LABELLED dataset
merged = data_lib.MergedDataset.from_folder_of_folders(
    base_dir=labeled_data_path,
    embedding_model=embedding_model,
    time_pooling=time_pooling,
    load_audio=False,
    target_sample_rate=-2,
    audio_file_pattern='*'
)

# Label distribution
lbl_counts = np.sum(merged.data['label_hot'], axis=0)
print('num classes :', (lbl_counts > 0).sum())
print('mean ex / class :', lbl_counts.sum() / (lbl_counts > 0).sum())
print('min ex / class :', (lbl_counts + (lbl_counts == 0) * 1e6).min())


#@title Train linear model over embeddings. { vertical-output: true }

# Number of random training examples to choose form each class.
# Set exactly one of train_ratio and train_examples_per_class
train_ratio = 0.95  #@param
train_examples_per_class = None  #@param

# Number of random re-trainings. Allows judging model stability.
num_seeds = 1  #@param

# Classifier training hyperparams.
# These should be good defaults.
batch_size = 32
num_epochs = 128
num_hiddens = -1
learning_rate = 1e-3

metrics = collections.defaultdict(list)
for seed in tqdm.tqdm(range(num_seeds)):
  if num_hiddens > 0:
    model = classify.get_two_layer_model(
        num_hiddens, merged.embedding_dim, merged.num_classes)
  else:
    model = classify.get_linear_model(
        merged.embedding_dim, merged.num_classes)
  run_metrics = classify.train_embedding_model(
      model, merged, train_ratio, train_examples_per_class,
      num_epochs, seed, batch_size, learning_rate)
  metrics['acc'].append(run_metrics.top1_accuracy)
  metrics['auc_roc'].append(run_metrics.auc_roc)
  metrics['cmap'].append(run_metrics.cmap_value)
  metrics['maps'].append(run_metrics.class_maps)
  metrics['test_logits'].append(run_metrics.test_logits)

mean_acc = np.mean(metrics['acc'])
mean_auc = np.mean(metrics['auc_roc'])
mean_cmap = np.mean(metrics['cmap'])
# Merge the test_logits into a single array.
test_logits = {
    k: np.concatenate([logits[k] for logits in metrics['test_logits']])
    for k in metrics['test_logits'][0].keys()
}

print(f'acc:{mean_acc:5.2f}, auc_roc:{mean_auc:5.2f}, cmap:{mean_cmap:5.2f}')
for lbl, auc in zip(merged.labels, run_metrics.class_maps):
  if np.isnan(auc):
    continue
  print(f'\n{lbl:8s}, auc_roc:{auc:5.2f}')
  colab_utils.prstats(f'test_logits({lbl})',
                      test_logits[merged.labels.index(lbl)])
  

  #@title Run model on target unlabeled data. { vertical-output: true }

# Choose the target class to work with.
target_class = 'pos'  #@param
# Choose a target logit; will display results close to the target.
# Set to None to get the highest-logit examples.
target_logit = None  #@param
# Number of results to display.
num_results = 25  #@param

# Create the embeddings dataset.
embeddings_ds = tf_examples.create_embeddings_dataset(
    embeddings_path, file_glob='embeddings-*')
target_class_idx = merged.labels.index(target_class)
results, all_logits = search.classifer_search_embeddings_parallel(
    embeddings_classifier=model,
    target_index=target_class_idx,
    embeddings_dataset=embeddings_ds,
    hop_size_s=embedding_hop_size_s,
    target_score=target_logit,
    top_k=num_results
)

# Plot the histogram of logits.
_, ys, _ = plt.hist(all_logits, bins=128, density=True)
plt.xlabel(f'{target_class} logit')
plt.ylabel('density')
# plt.yscale('log')
plt.plot([target_logit, target_logit], [0.0, np.max(ys)], 'r:')
plt.show()

#@title Display results for the target label. { vertical-output: true }

display_labels = merged.labels

extra_labels = []  #@param
for label in extra_labels:
  if label not in merged.labels:
    display_labels += (label,)
if 'unknown' not in merged.labels:
  display_labels += ('unknown',)

display.display_search_results(
    results, embedding_model.sample_rate,
    project_state.source_map,
    checkbox_labels=display_labels,
    max_workers=5)

#@title Add selected results to the labeled data. { vertical-output: true }

# results.write_labeled_data(
#     config.annotated_path, embedding_model.sample_rate)