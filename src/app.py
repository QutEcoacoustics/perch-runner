#!/usr/local/bin/python

"""
Entrypoint for processing a folder of audio files
"""

import argparse
from src.config import load_config
from src.embed import embed



def main():

    parser = argparse.ArgumentParser(description="Perch Runner: audio embedding and classification")
    parser.add_argument("--embed", nargs='?', const=True, default=None,
                        help="embedding output format(s), e.g. parquet, csv, parquet-columns. Use --embed with no value for default (parquet).")
    parser.add_argument("--classify", nargs='?', const=True, default=None,
                        help="classification output format(s), e.g. parquet, csv. Use --classify with no value for default (parquet).")
    parser.add_argument("--source", default=None, help="path to the source audio folder")
    parser.add_argument("--output", default=None, help="path to the output folder")
    parser.add_argument("--config_file", default=None, help="path to the config file")
    parser.add_argument("--model_choice", default=None, help="model to use, e.g. perch_v2")
    parser.add_argument("--embedding_table_format", default=None, help="table format for embeddings, e.g. serialized, columns")
    parser.add_argument("--file_glob", default=None, help="glob pattern for audio files, e.g. '*/*', '*/*/*'. Auto-detected if not specified.")
    args = parser.parse_args()


    config = load_config(args.config_file, args)

    if config['embed']:
        embed(config)

    if config['classify']:
        print("classify is not implemented yet")


if __name__ == "__main__":
    main()