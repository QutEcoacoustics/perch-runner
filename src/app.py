#!/usr/local/bin/python

"""
Entrypoint for processing a single file
"""

import argparse
from src.config import load_config
from src.embed_audio_slim import embed_file_and_save, embed_folder
from src.inference_parquet import classify_file_and_save, process_folder
#import train_linear_model
#import inference_slim

def main():

    print("start")

    valid_commands = ('generate', 'train', 'classify')

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(valid_commands), help=" | ".join(valid_commands))
    parser.add_argument("--source_file", help="path to the file to analyze")
    parser.add_argument("--source_folder", help="path to the a folder of files to analyze")
    parser.add_argument("--config_file", default=None, help="path to the config file")
    parser.add_argument("--output_folder", help="where to save the result file")
    args = parser.parse_args()

    if bool(args.source_file) == bool(args.source_folder):
        parser.error('You must specify exactly one of --source_file or --source_folder, not both.')


    config = load_config(args.config_file)

    if args.source_file:

        if args.command == "generate":
            embed_file_and_save(args.source_file, args.output_folder, config)
        elif args.command == "train":
            parser.error('Incompatible args. Please specify --source_folder, not --source_file.')
        elif args.command == "classify":
            print(f"classify file {args.source_file} to {args.output_folder}")
            classify_file_and_save(args.source_file, args.output_folder, config)
        else:
            print("invalid command")

    else:

        if args.command == "generate":
            embed_folder(args.source_folder, args.output_folder, config)
        elif args.command == "train":
            # train_linear_model.train(args.source_file, config, args.output_folder)
            print("train: not implemented yet")
        elif args.command == "classify":
            print(f"classify folder {args.source_folder} to {args.output_folder}")
            process_folder(args.source_folder, args.output_folder, config)
        else:
            print("invalid command")


if __name__ == "__main__":
    main()