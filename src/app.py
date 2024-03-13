#!/usr/local/bin/python

"""
utils for parsing configuration from yml
"""

import argparse
from src.config import load_config
from src.embed_audio_slim import embed_file_and_save
from src.inference_parquet import classify_file_and_save
#import train_linear_model
#import inference_slim

def main():

    valid_commands = ('generate', 'train', 'inference')

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(valid_commands), help=" | ".join(valid_commands))
    parser.add_argument("--source_file", help="path to the file to analyze")
    parser.add_argument("--config_file", default=None, help="path to the config file")
    parser.add_argument("--output_folder", help="where to save the result file")
    args = parser.parse_args()

    config = load_config(args.config_file)

    if args.command == "generate":
        embed_file_and_save(args.source_file, args.output_folder, config)
    elif args.command == "train":
        # train_linear_model.train(args.source_file, config, args.output_folder)
        print("train")
    elif args.command == "classify":
        # inference_slim.analze
        classify_file_and_save(args.source_file, args.output_folder, config)
    else:
        print("invalid command")


if __name__ == "__main__":
    main()