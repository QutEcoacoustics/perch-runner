#!/usr/local/bin/python

"""
utils for parsing configuration from yml
"""

# import importlib
# importlib.reload(src.embed_audio_slim)

import argparse

from src.embed_audio_slim import embed_file_and_save
from src.config import parse_config
#import train_linear_model
#import inference_slim

def main():

    valid_commands = ('generate', 'train', 'inference')

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(valid_commands), help=" | ".join(valid_commands))
    parser.add_argument("--source_file", help="path to the file to analyze")
    parser.add_argument("--config_file", default=None, help="path to the config file")
    parser.add_argument("--output_file", help="where to save the result file")
    args = parser.parse_args()

    config = parse_config(args.config_file)

    if args.command == "generate":
        embed_file_and_save(args.source_file, args.output_file, config)
    elif args.command == "train":
        # train_linear_model.train(args.source_file, config, args.output_folder)
        print("train")
    elif args.command == "inference":
        # inference_slim.analze
        print("inference")
    else:
        print("invalid command")
