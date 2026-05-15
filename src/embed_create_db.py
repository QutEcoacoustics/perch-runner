import logging
import time
from pathlib import Path

import soundfile as sf
from ml_collections import config_dict
from perch_hoplite.agile import embed as agile_embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.zoo import model_configs

from src.resources import compute_workers, log_ram

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg'}


def _discover_audio_files(source: Path) -> list[Path]:
    """Return all audio files under source, recursively."""
    return [
        p for p in source.rglob('*')
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def _scan_audio_files(source: Path, file_glob: str, discovered_audio_files: list[Path] | None = None) -> float:
    """Scan audio files matching the glob and report stats.
    This is purely for logging, and is useful for e.g. estimating total duration for a given duration for walltime on HPC
    Returns total duration in seconds.
    """
    if discovered_audio_files is None:
        files = sorted(source.glob(file_glob))
        audio_files = [f for f in files if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
    else:
        # Auto-detected patterns are depth-only globs like *, */*, */*/*.
        target_depth = len([part for part in file_glob.split('/') if part == '*'])
        audio_files = sorted(
            [f for f in discovered_audio_files if len(f.relative_to(source).parts) == target_depth]
        )

    if not audio_files:
        log.warning("No audio files found matching glob '%s' under %s", file_glob, source)
        return 0.0

    durations = []
    for f in audio_files:
        try:
            info = sf.info(str(f))
            durations.append(info.duration)
        except sf.SoundFileRuntimeError as exc:
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


def _detect_glob_pattern(source: Path, discovered_audio_files: list[Path] | None = None) -> str:
    """Auto-detect a glob pattern from the shallowest discovered audio file.

    Walks the source tree and finds all audio files. Returns a glob pattern
    matching the shallowest depth found:
    - Top-level file -> '*'
    - One level deep  -> '*/*'
    - Two levels deep -> '*/*/*'
    etc.

    If deeper audio files exist, a warning is logged because auto-detection
    will only include files at the shallowest depth.
    """
    audio_files = discovered_audio_files if discovered_audio_files is not None else _discover_audio_files(source)

    depths = [len(audio_file.relative_to(source).parts) for audio_file in audio_files]

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


def create_database(config: dict):
    source = Path(config['source'])
    output = Path(config['output'])
    db_path_value = config.get('db_path', 'db')
    db_path_candidate = Path(db_path_value)
    db_path = db_path_candidate if db_path_candidate.is_absolute() else output / db_path_candidate

    use_file_sharding = True
    shard_length_in_seconds = 60
    dataset_name = config['dataset_name']

    # Use configured glob pattern, or auto-detect from file depth.
    configured_file_glob = config.get('file_glob')
    discovered_audio_files = None
    if configured_file_glob:
        file_glob = configured_file_glob
    else:
        # we do source file discovery so we can report stats and so we can auto-detect a glob pattern that matches the shallowest files
        # perch-hoplite then re-discovers the files based on the glob. 
        # There is a potential future feature in perch-hoplite to accept a list of files directly, which would eliminate the redundant discovery. 
        discovered_audio_files = _discover_audio_files(source)
        file_glob = _detect_glob_pattern(source, discovered_audio_files=discovered_audio_files)

    # Log audio stats and compute workers BEFORE touching the DB,
    # so this info is visible before any perch_hoplite output.
    log.info("Audio source: base_path=%s, file_glob=%s, shard_len_s=%s",
             source, file_glob, shard_length_in_seconds if use_file_sharding else None)
    total_duration = _scan_audio_files(
        source,
        file_glob,
        discovered_audio_files=discovered_audio_files,
    )
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
