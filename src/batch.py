#!/usr/local/bin/python

"""
Entrypoint for processing a csv of input and output files
"""

import argparse
import yaml
import csv
from pathlib import Path

from src.embed_audio_slim import embed_file_and_save
from src.config import load_config
#import train_linear_model
#import inference_slim

def read_items(source_csv, start_row, end_row):
    """
    given a path to a csv of items to analyze, and the rows to include, 
    returns a list of dicts that has the source and output paths
    """
     
    # Initialize an empty list to store the dictionaries
    items = []

    # Open the CSV file
    with open(source_csv, mode='r', encoding='utf-8') as file:
        # Create a csv.DictReader object
        reader = csv.DictReader(file)

        # Iterate over the CSV rows
        for row in reader:
            # Each row is a dictionary
            items.append(row)

    if start_row is None:
        start_row = 0

    if end_row is None:
        end_row = len(items) - 1

    # the +1 is because end_row is inclusive
    items = items[start_row:(end_row + 1)]

    return items


def batch(command, source_csv, start_row, end_row, config_file, overwrite_existing=False):

    items = read_items(source_csv, start_row, end_row)
    config = load_config(config_file, command)

    for item in items:

        if Path(item['output']).exists() or overwrite_existing:
            print(f"skipping {item['source']} because output already exists")
        else:
            print(f"processing {item['source']}")

            match command:
                case "generate":
                    embed_file_and_save(item['source'], item['output'], config)
                case "inference":
                    # inference_slim.analze
                    print("inference")
                case _:
                    print("invalid command")


def main ():
    """Just the arg parsing from command line"""
    valid_commands = ('generate', 'inference')
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(valid_commands), help=" | ".join(valid_commands))
    parser.add_argument("--source_csv", help="path to a csv that has the columns 'source' and 'output'")
    parser.add_argument("--start_row", default=None, help="which row on the csv to start from (zero index)")
    parser.add_argument("--end_row", help="last row in the csv to process (zero index)")
    parser.add_argument("--config_file", default=None, help="path to the config file")
    parser.add_argument("--overwrite_existing", default=False, help="if true, will overwrite existing files, else will skip if exists")
    args = parser.parse_args()
    batch(args.command, args.source_csv, int(args.start_row), int(args.end_row), args.config_file, args.overwrite_existing)


if __name__ == "__main__":
    main()



