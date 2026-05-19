"""Logging configuration for perch-runner.

Controls three independent log channels:

1. **perch-runner** (``src.*`` loggers) — our application code.
2. **perch-hoplite / root** — third-party library output (SQL traces, etc.).
3. **TensorFlow C++** — XLA compilation messages, allocation warnings, etc.
   Controlled via the ``TF_CPP_MIN_LOG_LEVEL`` environment variable.

Each channel can be set to a standard Python log level
(DEBUG, INFO, WARNING, ERROR, CRITICAL) via CLI flags or config file keys.

An optional ``--log_file`` sends **all** output (both channels) to a file
in addition to the console.
"""

import logging
import os

VALID_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Mapping from our level names to TF_CPP_MIN_LOG_LEVEL values
_TF_LEVEL_MAP = {
    'DEBUG': '0',
    'INFO': '0',
    'WARNING': '1',
    'ERROR': '2',
    'CRITICAL': '3',
}


def setup_logging(config: dict) -> None:
    """Configure logging from a resolved config dict.

    Expected keys (all optional, with defaults):
        log_level:          Level for src.* loggers       (default: INFO)
        hoplite_log_level:  Level for root / 3rd-party    (default: WARNING)
        tf_log_level:       Level for TF C++ output       (default: WARNING)
        log_file:           Path to a log file, or None   (default: None)
    """
    app_level = _resolve_level(config.get('log_level', 'INFO'))
    hoplite_level = _resolve_level(config.get('hoplite_log_level', 'WARNING'))
    tf_level = config.get('tf_log_level', 'WARNING').upper()

    # TF C++ logging is controlled via environment variable, must be set
    # before TF is imported (we set it early in app.py, update here).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = _TF_LEVEL_MAP.get(tf_level, '1')

    # Root logger — controls perch_hoplite and any other library.
    root = logging.getLogger()
    root.setLevel(hoplite_level)

    # If basicConfig hasn't been called yet (no handlers), add a console handler.
    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        root.addHandler(console)
    else:
        # Update the existing handler's format just in case.
        for h in root.handlers:
            h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

    # Our application loggers — independent level.
    logging.getLogger('src').setLevel(app_level)

    # Optional file handler — captures everything from both channels.
    log_file = config.get('log_file')
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        fh.setLevel(min(app_level, hoplite_level))
        root.addHandler(fh)


def _resolve_level(value) -> int:
    """Convert a level name string to a logging constant."""
    if isinstance(value, int):
        return value
    name = str(value).upper()
    if name not in VALID_LEVELS:
        raise ValueError(
            f"Invalid log level: {value!r}. Choose from {VALID_LEVELS}")
    return getattr(logging, name)
