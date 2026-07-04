"""Run the main embedding pipeline for one normalized config.

This module coordinates database creation, optional embedding-table export,
optional recognizer execution, and cleanup of temporary database outputs. It is
the main implementation behind the app-level `embed(config)` entrypoint.
"""

import logging
import time
import shutil
from pathlib import Path

from src.config import config_to_json
from src.embed_create_db import _detect_glob_pattern, _scan_audio_files, create_database
from src.db_to_table import run_recognizers_over_db, export_embeddings_table
from src.resources import log_ram
from src.version import PERCH_HOPLITE_VERSION, __version__

log = logging.getLogger(__name__)



def _make_export_metadata(config: dict) -> dict[str, str]:
    """Build parquet footer metadata for embedding export outputs.

    This helper is called only by `embed()` immediately before
    `export_embeddings_table(...)` is invoked.

    Why it exists:
    - Keep metadata construction in one place.
    - Include runtime and config provenance in exported parquet files.

    What it returns:
    - `perch_runner.version`: runner package version.
    - `perch_hoplite.version`: hoplite dependency version.
    - `perch_runner.config_json`: normalized JSON snapshot of effective config.
    """
    return {
        "perch_runner.version": __version__,
        "perch_hoplite.version": PERCH_HOPLITE_VERSION,
        "perch_runner.config_json": config_to_json(config, sort_keys=True),
    }


def _select_classify_filetype(config: dict) -> str:
    """Choose classifier output extension from config.

    Called by `embed()` when recognizers are configured.

    Behavior:
    - If `config["classify"]` includes `"parquet"`, return `"parquet"`.
    - Otherwise return `"csv"`.

    Why this helper exists:
    - Keep output-filetype selection logic in one function.
    - Avoid spreading ad-hoc checks for `classify` values in the main pipeline.
    """
    classify_formats = set(config.get("classify") or [])
    if "parquet" in classify_formats:
        return "parquet"
    return "csv"


def embed(config: dict):
    t_start = time.monotonic()

    log.info("Starting embedding: source=%s, output=%s, model=%s",
             config['source'], config['output'], config['model_choice'])
    log_ram()

    output_root = Path(config['output'])
    db_path = Path(config['db_path'])
    
    # Resolve db_path relative to output if not absolute
    if not db_path.is_absolute():
        db_path = output_root / db_path

    # Track if DB existed before this run
    db_existed_before = db_path.exists()

    try:
        audio_duration_s = create_database(config)
    except Exception:
        log.exception("ERROR: Embedding failed")
        raise

    embed_formats = config['embed']  # list of EmbeddingsFormat
    save_db = config.get('save_db', False)

    if embed_formats:
        log.info("Exporting embeddings to files (%s)...", 
                 ", ".join(f"{ef.filetype}/{ef.table_format}" for ef in embed_formats))
        export_embeddings_table(
            db_path=db_path,
            output_path=output_root,
            embeddings_formats=embed_formats,
            output_template=config.get('embeddings_output_path_template'),
            parquet_metadata=_make_export_metadata(config),
        )

    recognizers = config.get("recognizers", [])
    if recognizers:
        classify_filetype = _select_classify_filetype(config)
        log.info(
            "Running recognizers (%d config(s), output=%s)...",
            len(recognizers),
            classify_filetype,
        )
        run_recognizers_over_db(
            db_path=db_path,
            output_parent=output_root,
            recognizers=recognizers,
            classify_filetype=classify_filetype,
            output_template=(
                config.get('classify_output_path_template')
            ),
        )

    # Clean up database only if: (1) save_db is false AND (2) DB was created by this run (didn't exist before)
    if not save_db and not db_existed_before and db_path.exists():
        log.info("Cleaning up database at %s (save_db is false, DB was created by this run)", db_path)
        shutil.rmtree(db_path, ignore_errors=True)
    elif not save_db and db_existed_before:
        log.info("Preserving pre-existing database at %s (it existed before this run)", db_path)

    elapsed = time.monotonic() - t_start
    log_ram()
    if audio_duration_s > 0:
        audio_hours = audio_duration_s / 3600
        time_per_hour = elapsed / audio_hours
        log.info("Done. Total time: %.1fs (%.1f min) — %.1fs per hour of audio",
                 elapsed, elapsed / 60, time_per_hour)
    else:
        log.info("Done. Total time: %.1fs (%.1f min)", elapsed, elapsed / 60)

