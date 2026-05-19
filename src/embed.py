import logging
import time
import shutil
from pathlib import Path

from src.config import config_to_json
from src.embed_create_db import _detect_glob_pattern, _scan_audio_files, create_database
from src.embed_export_table import export_embeddings_table
from src.resources import log_ram
from src.version import PERCH_HOPLITE_VERSION, __version__

log = logging.getLogger(__name__)



def _make_export_metadata(config: dict) -> dict[str, str]:
    """Build parquet file-level metadata for exported embeddings."""
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

