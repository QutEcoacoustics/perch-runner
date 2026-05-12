#!/usr/local/bin/python

"""
Entrypoint for processing a folder of audio files
"""

import argparse
import logging
import os

# Limit TF C++ logging before TF is imported (overridden by setup_logging later)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

from src.config import load_config
from src.logging_config import setup_logging
from src.version import __version__, MODELS, PERCH_HOPLITE_VERSION


def embed(config):
    # lazy load heavy stuff only when needed
    from src.embed import embed as run_embed
    return run_embed(config)



def main():

    parser = argparse.ArgumentParser(description="Perch Runner: audio embedding and classification")
    parser.add_argument("command", nargs='?', default=None,
                        help="Optional command: 'version' to print version and exit.")
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
    parser.add_argument("--workers", default=None, help="number of worker threads for embedding, or 'auto' (default) to choose based on available RAM.")
    parser.add_argument("--log_level", default=None, help="log level for perch-runner output: DEBUG, INFO, WARNING, ERROR (default: INFO)")
    parser.add_argument("--hoplite_log_level", default=None, help="log level for perch-hoplite / library output: DEBUG, INFO, WARNING, ERROR (default: WARNING)")
    parser.add_argument("--tf_log_level", default=None, help="log level for TensorFlow C++ output: DEBUG, INFO, WARNING, ERROR (default: WARNING)")
    parser.add_argument("--log_file", default=None, help="path to a log file. Output is sent to both console and file.")
    args = parser.parse_args()

    if args.command == "version":
        print(f"perch-runner {__version__}")
        print(f"perch-hoplite {PERCH_HOPLITE_VERSION}")
        print("Models:")
        for name, info in MODELS.items():
            print(f"  {name}: {info['kaggle']} v{info['version']} ({info['embedding_dim']}d)")
        return

    try:
        config = load_config(args.config_file, args)
        setup_logging(config)
        log = logging.getLogger(__name__)

        log.info("Starting perch-runner version %s", __version__)

        if config['embed']:
            log.info("Embed requested using model: %s", config['model_choice'])
            embed(config)

        if config['classify']:
            log.info("Classify requested using model: %s", config['model_choice'])
            log.warning("classify is not implemented yet")
    except MemoryError:
        logging.getLogger(__name__).error(
            "OUT OF MEMORY: Not enough RAM to complete embedding. "
            "Try reducing --workers or increasing container memory.")
        raise SystemExit(137)
    except (FileNotFoundError, ValueError):
        # Let config errors propagate as-is (will be caught by Python)
        raise
    except Exception:
        logging.getLogger(__name__).exception("Fatal error")
        raise SystemExit(1)


if __name__ == "__main__":
    main()