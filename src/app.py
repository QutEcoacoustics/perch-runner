#!/usr/local/bin/python

"""
utils for parsing configuration from yml
"""

import argparse
import yaml
from ml_collections import config_dict

from embed_audio_slim import embed_one_file
#import train_linear_model
#import inference_slim


def parse_config(yml_source):

    with open(yml_source, 'r') as f:
        config = yaml.safe_load(f)

    return config_dict.create(**config)



parser = argparse.ArgumentParser()
parser.add_argument("--command", help="generate | train | inference")
parser.add_argument("--source_file", help="path to the file to analyze")
parser.add_argument("--config_file", help="to the config file")
parser.add_argument("--output_folder", help="file to embeddings to")
args = parser.parse_args()

config = parse_config(args.config_file)

match args.command:
    case "generate":
         embed_one_file(args.source_file, config, args.output_folder)
    case "train":
         # train_linear_model.train(args.source_file, config, args.output_folder)
         print("train")
    case "inference":
         # inference_slim.analze
         print("inference")
    case _:
        print("invalid command")