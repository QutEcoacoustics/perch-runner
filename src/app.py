#!/usr/local/bin/python

"""
utils for parsing configuration from yml
"""

import argparse

from src.embed_audio_slim import embed_file_and_save
from src.config import parse_config
#import train_linear_model
#import inference_slim


parser = argparse.ArgumentParser()
parser.add_argument("command", help="generate | train | inference")
parser.add_argument("--source_file", help="path to the file to analyze")
parser.add_argument("--config_file", default=None, help="path to the config file")
parser.add_argument("--output_file", help="where to save the result file")
args = parser.parse_args()

config = parse_config(args.config_file)

match args.command:
    case "generate":
         embed_file_and_save(args.source_file, args.output_file, config)
    case "train":
         # train_linear_model.train(args.source_file, config, args.output_folder)
         print("train")
    case "inference":
         # inference_slim.analze
         print("inference")
    case _:
        print("invalid command")