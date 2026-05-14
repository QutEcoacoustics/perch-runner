import logging
import time

from perch_hoplite.agile import source_info
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from perch_hoplite.agile import embed as agile_embed
from ml_collections import config_dict
from perch_hoplite.db import db_loader
from perch_hoplite.zoo import model_configs
from perch_hoplite.db import sqlite_usearch_impl
from src import data_frames
from src.resources import compute_workers, log_ram

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg'}


def _scan_audio_files(source: Path, file_glob: str) -> float:
    """Scan audio files matching the glob and report stats.

    Returns total duration in seconds.
    """
    files = sorted(source.glob(file_glob))
    audio_files = [f for f in files if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]

    if not audio_files:
        log.warning("No audio files found matching glob '%s' under %s", file_glob, source)
        return 0.0

    durations = []
    for f in audio_files:
        try:
            info = sf.info(str(f))
            durations.append(info.duration)
        except Exception as exc:
            log.warning("Could not read %s: %s", f.name, exc)
            durations.append(0.0)

    total = sum(durations)
    avg = total / len(durations) if durations else 0.0

    log.info("Audio files found: %d", len(audio_files))
    for f, dur in zip(audio_files, durations):
        log.info("  %s  (%.1fs)", f.name, dur)
    log.info("Total duration: %.1fs (%.1f min)", total, total / 60)
    log.info("Average duration: %.1fs", avg)

    return total



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

    # Determine which parquet table formats are requested
    parquet_formats = [ef for ef in embed_formats if ef.filetype == 'parquet']
    as_serialized = any(ef.table_format == 'serialized' for ef in parquet_formats)
    as_columns = any(ef.table_format == 'columns' for ef in parquet_formats)

    if parquet_formats:
        log.info("Exporting embeddings to parquet...")
        export_as_parquet(
            db_path=Path(config['output']) / 'hoplite',
            output_path=Path(config['output']) / 'embeddings',
            as_serialized=as_serialized,
            as_columns=as_columns,
        )

    # Check if hoplite DB should be kept
    keep_hoplite = any(ef.filetype == 'hoplite' for ef in embed_formats)
    if not keep_hoplite:
        import shutil
        shutil.rmtree(Path(config['output']) / 'hoplite', ignore_errors=True)

    elapsed = time.monotonic() - t_start
    log_ram()
    if audio_duration_s > 0:
        audio_hours = audio_duration_s / 3600
        time_per_hour = elapsed / audio_hours
        log.info("Done. Total time: %.1fs (%.1f min) — %.1fs per hour of audio",
                 elapsed, elapsed / 60, time_per_hour)
    else:
        log.info("Done. Total time: %.1fs (%.1f min)", elapsed, elapsed / 60)
     


def _detect_glob_pattern(source: Path) -> str:
    """Auto-detect a glob pattern from the shallowest discovered audio file.

    Walks the source tree and finds all audio files. Returns a glob pattern
    matching the shallowest depth found:
    - Top-level file → '*'
    - One level deep  → '*/*'
    - Two levels deep → '*/*/*'
    etc.

    If deeper audio files exist, a warning is logged because auto-detection
    will only include files at the shallowest depth.
    """
    depths = []
    for audio_file in source.rglob('*'):
        if audio_file.is_file() and audio_file.suffix.lower() in AUDIO_EXTENSIONS:
            rel = audio_file.relative_to(source)
            depths.append(len(rel.parts))

    if not depths:
        raise FileNotFoundError(f"No audio files found in {source}")

    shallowest_depth = min(depths)
    skipped = sum(1 for depth in depths if depth > shallowest_depth)
    pattern = '/'.join(['*'] * shallowest_depth)

    if skipped > 0:
        log.warning(
            "Auto-detected file_glob '%s' from shallowest audio depth; %d deeper audio file(s) will be skipped. "
            "Set --file_glob explicitly to include nested files.",
            pattern,
            skipped,
        )

    return pattern


def create_database(
        config: dict
):

    source = Path(config['source'])
    output = Path(config['output'])

    db_path = output / 'hoplite'

    use_file_sharding = True
    shard_length_in_seconds = 60
    dataset_name = config['dataset_name']

    # Use configured glob pattern, or auto-detect from file depth.
    file_glob = config.get('file_glob') or _detect_glob_pattern(source)

    # Log audio stats and compute workers BEFORE touching the DB,
    # so this info is visible before any perch_hoplite output.
    log.info("Audio source: base_path=%s, file_glob=%s, shard_len_s=%s",
             source, file_glob, shard_length_in_seconds if use_file_sharding else None)
    total_duration = _scan_audio_files(source, file_glob)
    num_workers = compute_workers(config.get('workers', 'auto'))
    log_ram()

    model_config_key = config['model_choice']
    # extract single model from list (deterministic due to sorted order)
    if isinstance(model_config_key, list):
        model_config_key = model_config_key[0]
    log.info("Using embedding model: %s", model_config_key)
    preset_info = model_configs.get_preset_model_config(model_config_key)

    db_config = config_dict.ConfigDict({
        'db_path': db_path,
    })

    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(
        preset_info.embedding_dim
    )
    db_config.usearch_cfg = usearch_cfg

    db_key = 'sqlite_usearch'
    db_config = db_loader.DBConfig(db_key, db_config)
    model_config = agile_embed.ModelConfig(
        model_key=preset_info.model_key,
        embedding_dim=preset_info.embedding_dim,
        model_config=preset_info.model_config,
    )

    # SQL trace noise is filtered by _SqlFilter in app.py
    db = db_config.load_db()

    audio_glob = source_info.AudioSourceConfig(
        dataset_name=dataset_name,
        base_path=str(source),
        file_glob=file_glob,
        min_audio_len_s=1.0,
        target_sample_rate_hz=-2,
        shard_len_s=float(shard_length_in_seconds) if use_file_sharding else None,
    )

    audio_sources = source_info.AudioSources((audio_glob,))

    worker = agile_embed.EmbedWorker(
        audio_sources=audio_sources,
        db=db,
        model_config=model_config,
        audio_worker_threads=num_workers,
    )

    t0 = time.monotonic()
    log.info("Starting model inference...")
    worker.process_all(target_dataset_name=dataset_name)
    elapsed = time.monotonic() - t0
    log.info("Model inference finished in %.1fs", elapsed)
    log_ram()

    windows = db.get_all_windows()
    count = len(windows)
    if count == 0:
        log.warning("WARNING: 0 embeddings were produced! "
                    "Check that audio files match glob '%s' under %s "
                    "and are longer than 1 second.", file_glob, source)
    else:
        log.info("Created %d embeddings in %s", count, db_path)
    return total_duration


def export_as_parquet(
        db_path: str | Path = "/mnt/output/hoplite",
        output_path: str | Path = "/mnt/output/embeddings",
        sourcemap=None,
        as_serialized=True,
        as_columns=False
):
    """Export embeddings from a perch-hoplite database to parquet files.

    Produces one parquet file per unique source (recording filename).

    Args:
        db_path: Path to the hoplite database directory.
        output_path: Directory to write parquet files to.
        sourcemap: Optional function mapping a source filename to an output
            parquet path (relative to output_path). If None, the recording
            filename is used with its extension replaced by .parquet.
        as_serialized: If True, embeddings are stored as a single base64-encoded
            column named 'embeddings'.
        as_columns: If True, each embedding dimension is stored in a separate
            column (f0000, f0001, ...) for backward compatibility.
    """
    if sourcemap is None:
        sourcemap = lambda x: Path(x) / 'embeddings.parquet'

    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path))

    # Build a mapping from recording_id to filename.
    recordings = {r.id: r.filename for r in db.get_all_recordings()}

    # Gather all windows and their embeddings.
    windows = db.get_all_windows()
    if not windows:
        log.warning("No embeddings found in database — nothing to export.")
        return

    window_ids = [w.id for w in windows]
    log.info("Reading %d embeddings from database...", len(window_ids))

    # usearch ≥2.25 returns a tuple of arrays from batch get() instead of a
    # 2-D ndarray, which makes perch_hoplite's get_embeddings_batch() crash.
    # Work around by fetching embeddings one-by-one.
    embeddings = [db.get_embedding(wid) for wid in window_ids]

    # Group raw data by source filename.
    data_by_source: dict[str, list[tuple[float, np.ndarray]]] = {}
    for window, embedding in zip(windows, embeddings):
        source = recordings[window.recording_id]
        offset = window.offsets[0]
        data_by_source.setdefault(source, []).append((offset, embedding))

    log.info("Exporting %d source(s) to parquet...", len(data_by_source))
    both = as_serialized and as_columns

    for i, (source, entries) in enumerate(data_by_source.items(), 1):
        entries.sort(key=lambda x: x[0])  # sort by offset
        rel_path = sourcemap(source)

        if as_columns:
            dest_dir = (output_path / 'columns') if both else output_path
            col_names = data_frames.embedding_col_names(len(entries[0][1]))
            rows = []
            for offset, embedding in entries:
                row = {'source': source, 'channel': 0, 'offset': offset}
                for name, val in zip(col_names, embedding):
                    row[name] = float(val)
                rows.append(row)
            df = pd.DataFrame(rows)
            dest = dest_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dest, index=False)
            log.info("  [%d/%d] Wrote %s (%d rows, columns format)",
                     i, len(data_by_source), dest, len(rows))

        if as_serialized:
            dest_dir = (output_path / 'serialized') if both else output_path
            rows = []
            for offset, embedding in entries:
                emb_array = np.asarray(embedding)
                encoded = data_frames.serialize_array(emb_array, dtype=emb_array.dtype)
                rows.append({
                    'source': source,
                    'channel': 0,
                    'offset': offset,
                    'embeddings': encoded,
                })
            df = pd.DataFrame(rows)
            dest = dest_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dest, index=False)
            log.info("  [%d/%d] Wrote %s (%d rows, serialized format)",
                     i, len(data_by_source), dest, len(rows))

