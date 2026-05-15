import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from perch_hoplite.db import sqlite_usearch_impl

from src import data_frames
from src.config import (
    DEFAULT_EMBEDDINGS_OUTPUT_PATH_TEMPLATE,
    EmbeddingsFormat,
    ensure_output_path_within_root,
    render_embeddings_output_relative_path,
)

log = logging.getLogger(__name__)
PARQUET_COMPRESSION = "snappy"


def export_embeddings_table(
    db_path: str | Path,
    output_path: str | Path,
    embeddings_formats: list[EmbeddingsFormat],
    sourcemap=None,
    output_template=None,
):
    """Export embeddings from a perch-hoplite database to tabular files.

    Produces one file per unique source (recording filename), per format.
    Supports both parquet and CSV formats with serialized or column-based embeddings.

    Args:
        db_path: Path to the hoplite database directory.
        output_path: Directory to write files to.
        sourcemap: Optional function mapping a source filename to an output
            source value written into the rows. If None, the recording
            filename is used unchanged.
        output_template: Optional relative path template that determines the
            destination. If None, the default embeddings template is used.
        embeddings_formats: List of EmbeddingsFormat objects specifying which
            combinations of filetype (csv/parquet) and table_format (serialized/columns)
            to export.
    """

    if sourcemap is None:
        sourcemap = lambda x: x

    if output_template is None:
        output_template = DEFAULT_EMBEDDINGS_OUTPUT_PATH_TEMPLATE

    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path))

    # Build a mapping from recording_id to filename.
    recordings = {r.id: r.filename for r in db.get_all_recordings()}

    # Gather all windows.
    windows = db.get_all_windows()
    if not windows:
        log.warning("No embeddings found in database - nothing to export.")
        return

    log.info("Preparing %d embedding reference(s) from database...", len(windows))

    # create a dict of {recording:  [(offset, window_id), ...]}
    data_by_source: dict[str, list[tuple[float, int]]] = {}
    for window in windows:
        source = recordings[window.recording_id]
        offset = window.offsets[0]
        data_by_source.setdefault(source, []).append((offset, window.id))

    log.info("Exporting %d source(s) to %d format(s)...",
             len(data_by_source), len(embeddings_formats))

    def output_destination(source: str, embedding_table_format: str, filetype: str) -> Path:
        """tiny helper to get the output path and do some validation"""
        ext = f".{filetype}"
        # Use the path as stored in the DB directly (already relative)
        rel_path = render_embeddings_output_relative_path(
            template=output_template,
            audio_file=source,
            output_ext=ext,
            embedding_table_format=embedding_table_format,
            analysis="embed",
        )
        return ensure_output_path_within_root(rel_path, output_path)

    for embeddings_format in embeddings_formats:
        filetype = embeddings_format.filetype
        table_format = embeddings_format.table_format

        # Precompute all destinations so we can clear stale in-progress files
        # before any new writes begin.

        # map sources to destinations for this format (each format can have a different destination due to {embedding_table_format} token and file extension)
        dest_paths_map = {source: output_destination(source, table_format, filetype) for source in data_by_source}

        dest_paths_set = set(dest_paths_map.values())

        # remove any in-progress files that might be left over from a previous failed run, to avoid appending to them by mistake. 
        # This means we can't resume failed runs, but it's simpler than dealing with partial files. 
        for dest in dest_paths_set:
            dest_inprogress = dest.with_suffix(dest.suffix + '.inprogress')
            if dest_inprogress.exists():
                dest_inprogress.unlink()
                log.info("Removed stale in-progress file %s", dest_inprogress)

        # as we iterate through source, build the set of in-progress files we create so we can finalize them at the end
        inprogress_paths = set()
        parquet_writers: dict[Path, pq.ParquetWriter] = {}

        for i, (source, entries) in enumerate(data_by_source.items(), 1):
            entries.sort(key=lambda x: x[0])
            output_source_value = sourcemap(source)

            dest = dest_paths_map[source]
            df = build_rows(output_source_value, entries, table_format, db)
            dest_inprogress = dest.with_suffix(dest.suffix + '.inprogress')
            dest.parent.mkdir(parents=True, exist_ok=True)

            if filetype == 'parquet':
                write_inprogress_parquet(dest_inprogress, df, parquet_writers)
            elif filetype == 'csv':
                write_inprogress_csv(dest_inprogress, df)
            else:
                raise ValueError(f"Unsupported filetype: {filetype}")

            inprogress_paths.add(dest_inprogress)
            log.info("  [%d/%d] Wrote %s (%d rows, %s/%s, inprogress)",
                     i, len(data_by_source), dest_inprogress.name, len(df), filetype, table_format)

        for writer in parquet_writers.values():
            writer.close()

        for inprogress_path in inprogress_paths:
            finalize_inprogress_file(inprogress_path, filetype)



def finalize_inprogress_file(inprogress_path: Path, filetype: str) -> None:
    """Finalize a .inprogress file by deduping, sorting, and writing final output."""

    try:
        if filetype == 'parquet':
            df = pd.read_parquet(inprogress_path)
        elif filetype == 'csv':
            df = pd.read_csv(inprogress_path)
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")

        df = df.drop_duplicates()
        sort_cols = [c for c in ['source', 'offset'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols)

        final_path = inprogress_path.with_suffix('')
        if filetype == 'parquet':
            df.to_parquet(final_path, index=False, compression=PARQUET_COMPRESSION)
        elif filetype == 'csv':
            df.to_csv(final_path, index=False)

        inprogress_path.unlink()
        log.info("Finalized %s (deduped, sorted, renamed)", final_path)
    except (OSError, ValueError) as e:
        log.error("Failed to finalize %s: %s", inprogress_path, e)


def build_rows(source_value, entries: list[tuple[float, int]], embedding_table_format, db) -> pd.DataFrame:
    """Build a DataFrame for one source by fetching embeddings by window id."""
    if embedding_table_format == 'columns':
        first_embedding = np.asarray(db.get_embedding(entries[0][1]))
        col_names = data_frames.embedding_col_names(len(first_embedding))
        rows = []
        for offset, embedding_id in entries:
            embedding = np.asarray(db.get_embedding(embedding_id))
            row = {'source': source_value, 'channel': 0, 'offset': offset}
            for name, val in zip(col_names, embedding):
                row[name] = float(val)
            rows.append(row)
        return pd.DataFrame(rows)

    rows = []
    for offset, embedding_id in entries:
        embedding = np.asarray(db.get_embedding(embedding_id))
        emb_array = np.asarray(embedding, dtype=np.float32)
        encoded = data_frames.serialize_array(emb_array, dtype=np.float32)
        rows.append({
            'source': source_value,
            'channel': 0,
            'offset': offset,
            'embeddings': encoded,
        })
    return pd.DataFrame(rows)

def write_inprogress_parquet(dest: Path, df: pd.DataFrame, writers: dict) -> None:
    """Append rows to a parquet in-progress file using a cached writer.

    Args:
        dest: Target `.inprogress` parquet path.
        df: Rows to append.
        writers: Mutable cache mapping in-progress file paths to open
            `pyarrow.parquet.ParquetWriter` instances, reused across writes.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    writer = writers.get(dest)
    if writer is None:
        writer = pq.ParquetWriter(dest, table.schema, compression=PARQUET_COMPRESSION)
        writers[dest] = writer
    writer.write_table(table)

def write_inprogress_csv(dest: Path, df: pd.DataFrame) -> None:
    """Append rows to a CSV in-progress file, writing header only once."""
    if dest.exists():
        df.to_csv(dest, mode='a', header=False, index=False)
    else:
        df.to_csv(dest, mode='w', header=True, index=False)
        