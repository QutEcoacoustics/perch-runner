# based off https://github.com/google-research/perch/blob/main/active_learning.ipynb

# Global imports
import collections
import json
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
from etils import epath
import matplotlib.pyplot as plt

from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

# Chirp imports
from chirp.models import metrics
from chirp.inference import tf_examples
from chirp.projects.bootstrap import bootstrap
from chirp.projects.bootstrap import display
from chirp.projects.bootstrap import search
from chirp.projects.multicluster import classify
from chirp.projects.multicluster import data_lib

# embeddings of unlabelled audio. 
# why is this needed?? basically just because there is a combined config object
# that holds info for both the embeddings and the labelled data path
embeddings_path = '/phil/output/pw_embeddings_all/'  #@param
embeddings_path = '/phil/output/site_039/'  #@param

#labeled_data_path = '/phil/plains_wanderer_labelled/sorted_no_raven/'  #@param
labeled_data_path = '/phil/plains_wanderer_labelled/tiny_sample/'  #@param

# Get relevant info from the embedding configuration.
embeddings_path = epath.Path(embeddings_path)
with (embeddings_path / 'config.json').open() as f:
  embedding_config = config_dict.ConfigDict(json.loads(f.read()))
embeddings_glob = embeddings_path / 'embeddings-*'
embedding_hop_size_s = embedding_config.embed_fn_config.model_config.hop_size_s

config = bootstrap.BootstrapConfig.load_from_embedding_config(
    embeddings_path=embeddings_path,
    annotated_path=labeled_data_path)

# this object holds 
# - the config 
# - the loaded model
# - the embeddings 'dataset'. I think this is a tf dataset of the unlabelled embeddings
# - a 'source map' which is a dict that maps the path to the original audio stored in the 
#   embedding metadata (the saved embedding file) to the full path to the audio.  
project_state = bootstrap.BootstrapState(config)
embedding_model = project_state.embedding_model


def supervised_learning():
    """
    This is based on everything under the Supervised Learning heading in the notebook
    """

    # Time-pooling strategy for examples longer than the model's window size.
    time_pooling = 'mean'  #@param

    # this performs the embeddings on the labelled set, so it takes a while
    # todo: save these to disk
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


    # Number of random training examples to choose form each class.
    train_ratio = 0.75
    train_examples_per_class = None  #@param

    # Number of random re-trainings. Allows judging model stability.
    num_seeds = 1  #@param

    # Classifier training hyperparams.
    # These should be good defaults.
    batch_size = 4
    #num_epochs = 128
    num_epochs = 8
    num_hiddens = -1
    learning_rate = 1e-3
    
    # creates a dictionary of lists where we can append to list by dictionary key
    # and if the key is not present yet it will be created with an empty list
    metrics2 = collections.defaultdict(list)
    for seed in range(num_seeds):
        if num_hiddens > 0:
            model = classify.get_two_layer_model(num_hiddens, merged.embedding_dim, merged.num_classes, True)
        else:
            model = classify.get_linear_model(merged.embedding_dim, merged.num_classes)

        run_metrics = classify.train_embedding_model(model, merged, train_ratio, train_examples_per_class, num_epochs, seed, batch_size, learning_rate)
        metrics2['acc'].append(run_metrics.top1_accuracy)
        metrics2['auc_roc'].append(run_metrics.auc_roc)
        metrics2['cmap'].append(run_metrics.cmap_value)
        metrics2['maps'].append(run_metrics.class_maps)
        metrics2['test_logits'].append(run_metrics.test_logits)
    mean_acc = np.mean(metrics2['acc'])
    mean_auc = np.mean(metrics2['auc_roc'])
    mean_cmap = np.mean(metrics2['cmap'])
    # Merge the test_logits into a single array.
    test_logits = {
        k: np.concatenate([logits[k] for logits in metrics2['test_logits']])
        for k in metrics2['test_logits'][0].keys()
    }

    print(f'acc:{mean_acc:5.2f}, auc_roc:{mean_auc:5.2f}, cmap:{mean_cmap:5.2f}')
    for lbl, auc in zip(merged.labels, run_metrics.class_maps):
      if np.isnan(auc):
        continue
      print(f'\n{lbl:8s}, auc_roc:{auc:5.2f}')
      colab_utils.prstats(f'test_logits({lbl})', test_logits[merged.labels.index(lbl)])

    return model



model = supervised_learning()
save_path = "/output/trained_model.keras"
# https://www.tensorflow.org/guide/keras/serialization_and_saving
model.save(save_path)
print(model)
