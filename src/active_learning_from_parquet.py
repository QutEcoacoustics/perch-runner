import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass

from chirp.projects.bootstrap import display
from chirp.projects.bootstrap.search import TopKSearchResults, SearchResult


from inference_parquet import process_embeddings
from train_linear_model_slim import train_and_save



def active_learning(labelled_source, model_output_file, embedding_model_version, embeddings_dir, search_results_file, skip_if_file_exists = False):

    if skip_if_file_exists and Path(model_output_file).exists() and Path(model_output_file + '.labels.json').exists():
        classifier_model = model_output_file
        with open(model_output_file + '.labels.json', 'r') as file:
            labels = json.load(file)
    else:
        classifier_model, labels = train_and_save(labelled_source, model_output_file, embedding_model_version)

    if skip_if_file_exists and Path(search_results_file).exists():
        inference_results = pd.read_csv(search_results_file)
    else:
        inference_results = process_embeddings(embeddings_dir, classifier_model, search_results_file, labels)

    display_search_results(inference_results, tuple(labels) + ('unknown',))




def plot_logits(inference_results, target_class, target_logit = None):

    all_logits = inference_results[target_class].values
    

    # Plot the histogram of logits.
    _, ys, _ = plt.hist(all_logits, bins=128, density=True)
    plt.xlabel(f'{target_class} logit')
    plt.ylabel('density')
    # plt.yscale('log')
    plt.plot([target_logit, target_logit], [0.0, np.max(ys)], 'r:')
    plt.show()

#@title Display results for the target label. { vertical-output: true }


def display_search_results(results_df, labels, target_label='pos', sample_rate = 32000):
    
    
    # class Result():
    #     def __init__(self, **kwargs):
    #         self.__dict__.update(kwargs)

    # results = [Result(**row) for row in results_df.to_dict('records')]


    results = TopKSearchResults([], top_k=20)

    for row in results_df.to_dict('records'):
        result = SearchResult(
            filename=row['filename'],
            timestamp_offset=row['offset_seconds'],
            embedding=None,
            score=row[target_label],
            sort_score=row[target_label]
        )
        results.update(result)
        
        
    display_labels = labels

    source_map = {source: source for source in results_df['filename'].unique()}


    extra_labels = []  #@param
    for label in extra_labels:
        if label not in labels:
            display_labels += (label,)
    if 'unknown' not in labels:
        display_labels += ('unknown',)

    display.display_search_results(
        results, sample_rate,
        source_map,
        checkbox_labels=display_labels,
        max_workers=5)
    
    print("done")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled_source", help="path to folder of folders")
    parser.add_argument("--model_output_file", default='/output/trained_model.keras', help="where to save the model")
    parser.add_argument("--embedding_model_version", default=4, help="path to embedding model")
    parser.add_argument("--embeddings_dir", help="path to directory of embeddings files")
    parser.add_argument("--search_results", help="save the results here")
    parser.add_argument("--skip_if_file_exists", default=0, type=int, help="if true, will not perform any step where the output file already exists")
    args = parser.parse_args()
    #config = config_dict.create(**vars(args))

    active_learning(args.labelled_source, args.model_output_file, int(args.embedding_model_version), args.embeddings_dir, args.search_results, bool(int(args.skip_if_file_exists)))