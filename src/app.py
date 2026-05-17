#!/usr/local/bin/python

"""
Entrypoint for processing a folder of audio files
"""

import argparse
import json
import logging
import os

# Limit TF C++ logging before TF is imported (overridden by setup_logging later)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

from src.config import default_config, load_config
from src.logging_config import setup_logging
from src.version import __version__, MODELS, PERCH_HOPLITE_VERSION


def embed(config):
    # lazy load heavy stuff only when needed
    from src.embed import embed as run_embed
    return run_embed(config)



def main():

    parser = argparse.ArgumentParser(description="Perch Runner: audio embedding and classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="run embedding/classification analysis",
    )
    analyze_parser.add_argument("--embed", nargs='?', const=True, default=None,
                                help="embedding output format(s), e.g. parquet, csv, parquet-columns. Use --embed with no value for default (parquet).")
    analyze_parser.add_argument("--classify", nargs='?', const=True, default=None,
                                help="classification output format(s), e.g. parquet, csv. Use --classify with no value for default (csv).")
    analyze_parser.add_argument("--source", default=None, help="path to the source audio folder")
    analyze_parser.add_argument("--output", default=None, help="path to the output folder")
    analyze_parser.add_argument("--config_file", default=None, help="path to the config file")
    analyze_parser.add_argument("--model_choice", default=None, help="model to use, e.g. perch_v2")
    analyze_parser.add_argument("--embedding_table_format", default=None, help="table format for embeddings, e.g. serialized, columns")
    analyze_parser.add_argument(
        "--embeddings_output_path_template",
        default=None,
        help=(
            "custom output path template for embeddings files. "
            "Supported tokens: {parents}, {basename}, {ext}, {embedding_table_format}, {analysis}."
        ),
    )
    analyze_parser.add_argument(
        "--embeddings_output_path_type",
        default=None,
        help="preset output path type: flat_basename, nested_basename, nested, flat",
    )
    analyze_parser.add_argument("--db_path", default=None, help="database output path. Relative paths are resolved under --output (default: db)")
    analyze_parser.add_argument("--file_glob", default=None, help="glob pattern for audio files, e.g. '*/*', '*/*/*'. Auto-detected if not specified.")
    analyze_parser.add_argument("--workers", default=None, help="number of worker threads for embedding, or 'auto' (default) to choose based on available RAM.")
    analyze_parser.add_argument("--log_level", default=None, help="log level for perch-runner output: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)")
    analyze_parser.add_argument("--hoplite_log_level", default=None, help="log level for perch-hoplite / library output: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)")
    analyze_parser.add_argument("--tf_log_level", default=None, help="log level for TensorFlow C++ output: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)")
    analyze_parser.add_argument("--log_file", default=None, help="path to a log file. Output is sent to both console and file.")

    subparsers.add_parser("version", help="print version and exit")
    subparsers.add_parser("config", help="print default config and exit")

    args = parser.parse_args()

    if args.command == "version":
        print(f"perch-runner {__version__}")
        print(f"perch-hoplite {PERCH_HOPLITE_VERSION}")
        print("Models:")
        for name, info in MODELS.items():
            print(f"  {name}: {info['kaggle']} v{info['version']} ({info['embedding_dim']}d)")
        return

    if args.command == "config":
        print(json.dumps(default_config, indent=2))
        return

    try:
        config_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != "command"})
        config = load_config(args.config_file, config_args)
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