from perch_hoplite.agile import source_info
from pathlib import Path
import numpy as np
import pandas as pd
from perch_hoplite.agile import embed as agile_embed
from ml_collections import config_dict
from perch_hoplite.db import db_loader
from perch_hoplite.zoo import model_configs
from perch_hoplite.db import sqlite_usearch_impl
from src import data_frames

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg'}



def embed(config: dict):

    create_database(config)

    embed_formats = config['embed']  # list of EmbeddingsFormat

    # Determine which parquet table formats are requested
    parquet_formats = [ef for ef in embed_formats if ef.filetype == 'parquet']
    as_serialized = any(ef.table_format == 'serialized' for ef in parquet_formats)
    as_columns = any(ef.table_format == 'columns' for ef in parquet_formats)

    if parquet_formats:
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
     


def _detect_glob_pattern(source: Path) -> str:
    """Auto-detect the glob pattern by finding the depth of the first audio file.

    Walks the source tree looking for audio files. Returns a glob pattern
    matching the depth of the first one found:
    - Top-level file → '*'
    - One level deep  → '*/*'
    - Two levels deep → '*/*/*'
    etc.
    """
    for audio_file in source.rglob('*'):
        if not audio_file.is_file():
            continue
        if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
            rel = audio_file.relative_to(source)
            depth = len(rel.parts)
            return '/'.join(['*'] * depth)

    raise FileNotFoundError(f"No audio files found in {source}")


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

    model_config_key = config['model_choice']
    if isinstance(model_config_key, set):
        model_config_key = next(iter(model_config_key))
    preset_info = model_configs.get_preset_model_config(model_config_key)

    db_config = config_dict.ConfigDict({
        'db_path': db_path,
    })

    db_config.usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(
        preset_info.embedding_dim
    )

    db_key = 'sqlite_usearch'
    db_config = db_loader.DBConfig(db_key, db_config)
    model_config = agile_embed.ModelConfig(
        model_key=preset_info.model_key,
        embedding_dim=preset_info.embedding_dim,
        model_config=preset_info.model_config,
    )

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
    )
    worker.process_all(target_dataset_name=dataset_name)

    return True


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
        return

    window_ids = [w.id for w in windows]

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

    both = as_serialized and as_columns

    for source, entries in data_by_source.items():
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

        if as_serialized:
            dest_dir = (output_path / 'serialized') if both else output_path
            rows = []
            for offset, embedding in entries:
                emb_array = np.asarray(embedding, dtype=np.float32)
                encoded = data_frames.serialize_array(emb_array)
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

