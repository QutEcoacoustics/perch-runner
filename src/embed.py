import logging
import time
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

    try:
        audio_duration_s = create_database(config)
    except Exception:
        log.exception("ERROR: Embedding failed")
        raise

    embed_formats = config['embed']  # list of EmbeddingsFormat

    output_root = Path(config['output'])
    db_path = Path(config['db_path'])

    if embed_formats:
        log.info("Exporting embeddings to files (%s)...", 
                 ", ".join(f"{ef.filetype}/{ef.table_format}" for ef in embed_formats))
        export_embeddings_table(
            db_path=db_path,
            output_path=output_root / 'embeddings',
            embeddings_formats=embed_formats,
            output_template=config.get('embeddings_output_path_template'),
            parquet_metadata=_make_export_metadata(config),
        )

    elapsed = time.monotonic() - t_start
    log_ram()
    if audio_duration_s > 0:
        audio_hours = audio_duration_s / 3600
        time_per_hour = elapsed / audio_hours
        log.info("Done. Total time: %.1fs (%.1f min) — %.1fs per hour of audio",
                 elapsed, elapsed / 60, time_per_hour)
    else:
        log.info("Done. Total time: %.1fs (%.1f min)", elapsed, elapsed / 60)

