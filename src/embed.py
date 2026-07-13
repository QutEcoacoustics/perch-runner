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
from src.embed_create_db import AUDIO_EXTENSIONS, _detect_glob_pattern, _discover_audio_files, create_database
from src.db_to_table import run_recognizers_over_db, export_embeddings_table
from src.resources import log_ram
from src.sourcemap import build_sourcemap
from src.version import PERCH_HOPLITE_VERSION, __version__

log = logging.getLogger(__name__)


def _count_input_audio_files_for_run(config: dict) -> int:
    """Estimate how many audio files will be processed for this run."""
    source = Path(config["source"])
    if source.is_file():
        return 1

    configured_file_glob = config.get("file_glob")
    if configured_file_glob:
        files = sorted(source.glob(configured_file_glob))
        return sum(1 for f in files if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)

    discovered_audio_files = _discover_audio_files(source)
    if not discovered_audio_files:
        return 0

    file_glob = _detect_glob_pattern(source, discovered_audio_files=discovered_audio_files)
    target_depth = len([part for part in file_glob.split("/") if part == "*"])
    return sum(
        1
        for f in discovered_audio_files
        if len(f.relative_to(source).parts) == target_depth
    )



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

    sourcemap_config = config.get("sourcemap_config")
    if sourcemap_config is not None and sourcemap_config.sourcemap_pattern is None:
        input_audio_count = _count_input_audio_files_for_run(config)
        if input_audio_count > 1:
            raise ValueError(
                "sourcemap without a pattern cannot be used when processing multiple audio files, "
                "because it would map all files to the same source value. "
                "Provide sourcemap_pattern or process a single input file."
            )

    try:
        audio_duration_s = create_database(config)
    except Exception:
        log.exception("ERROR: Embedding failed")
        raise

    
    save_db = config.get('save_db', False)
    sourcemap = build_sourcemap(sourcemap_config)

    if config['embed']:
        export_embeddings_table(
            db_path=db_path,
            output_path=output_root,
            table_format=config['embeddings_table_format'],
            filetype=config["embeddings_table_filetype"],
            output_template=config["embeddings_output_path_template"],
            sourcemap=sourcemap,
            parquet_metadata=_make_export_metadata(config),
        )

    recognizers = config.get("recognizers", [])
    if recognizers:
        run_recognizers_over_db(
            db_path=db_path,
            output_parent=output_root,
            recognizers=recognizers,
            recognizer_results_filetype=config["recognizer_results_filetype"],
            output_template=config['recognizer_output_path_template'],
            sourcemap=sourcemap,
            parquet_metadata=_make_export_metadata(config),
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

