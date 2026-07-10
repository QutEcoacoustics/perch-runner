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

from src.config import all_config_options, default_config, load_config
from src.logging_config import setup_logging
from src.version import __version__, MODELS, PERCH_HOPLITE_VERSION


def embed(config):
    # lazy load heavy stuff only when needed
    from src.embed import embed as run_embed
    return run_embed(config)


def handle_analyze(args):
    """Handle the analyze subcommand."""
    try:
        # Filter out argparse-specific fields (command, func) before passing to load_config
        config_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in ('command', 'func')})
        config = load_config(args.config_file, config_args)
        setup_logging(config)
        log = logging.getLogger(__name__)

        log.info("Starting perch-runner version %s", __version__)

        # there are two analyze branches: embedding (and then maybe doing something with the embeddings), or classify, which does not produce embeddings. 
        if config['embed'] or config['save_db'] or config.get('recognizers'):
            log.info("Embed requested using model: %s", config['model_choice'])
            embed(config)

        if config['classify']:
            # TODO: implement classify branch (no embedding)
            pass

    except MemoryError:
        logging.getLogger(__name__).error(
            "OUT OF MEMORY: Not enough RAM to complete embedding. "
            "Try reducing --workers or increasing container memory.")
        raise SystemExit(137)
    except (FileNotFoundError, ValueError):
        raise
    except Exception:
        logging.getLogger(__name__).exception("Fatal error")
        raise SystemExit(1)


def handle_version(args):
    """Handle the version subcommand."""
    print(f"perch-runner {__version__}")
    print(f"perch-hoplite {PERCH_HOPLITE_VERSION}")
    print("Models:")
    for name, info in MODELS.items():
        print(f"  {name}: {info['kaggle']} v{info['version']} ({info['embedding_dim']}d)")


def handle_config(args):
    """Handle the config subcommand."""
    print(json.dumps(default_config, indent=2))


def get_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Perch Runner: audio embedding and classification")
    subparsers = parser.add_subparsers(dest="command", required=True)


    analyze_parser = subparsers.add_parser(
        "analyze",
        help="run embedding/classification analysis",
    )

    analyze_parser.add_argument(
        "--config_file", 
        default=None, 
        help="path to the config file"
    )
    # 2. Dynamically generate arguments from your single source of truth
    # Don't apply defaults here. Config merging relies on non-specified items being absent. 
    for key, (_default_val, help_text) in all_config_options.items():
        kwargs = {
            "help": help_text,
            "default": argparse.SUPPRESS
        }
        
        # embed/classify/save_db support optional bool-like values.
        # Examples: --embed (True), --embed false (False), omitted (no override).
        if key in ["embed", "classify", "save_db"]:
            kwargs["nargs"] = '?'
            kwargs["const"] = True
            
        analyze_parser.add_argument(f"--{key}", **kwargs)

    analyze_parser.set_defaults(func=handle_analyze)

    version_parser = subparsers.add_parser("version", help="print version and exit")
    version_parser.set_defaults(func=handle_version)

    config_parser = subparsers.add_parser("config", help="print default config and exit")
    config_parser.set_defaults(func=handle_config)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()