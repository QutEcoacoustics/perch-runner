"""Export embeddings tables and recognizer outputs from a hoplite database.

This module reads the database produced by the embedding stage, groups rows by
source recording, writes embedding tables in CSV or parquet form, and runs
embeddings-based recognizers to produce per-recognizer output files.

The functionality writing embeddings to tables and running recognizers shares a lot of common logic.
That is the reason they are grouped together here. 
Things like
- loading the database
- grouping windows by source
- building a DataFrame for each source
- resolving the output path for each source (which might be shared by multiple sources if the output template does not include {basename})
- appending output to in-progress files
- cleaning up in-progress files by deduping, sorting, and renaming to final output
are all shared between the two operations.


Note on naming: in this app we use recognizer (and recognizer name etc) to refer to the embeddings-classifier linear classifiers, 
and classify to refer to the perch global base model classification. However, confusingly, recognizers are run using the "embeddings-classifier" package
which internally refers to them as "classifiers". So in this code there are references to recognizer_name and classifier_name, which are the same thing.

"""

import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from perch_hoplite.db import sqlite_usearch_impl


from src import data_frames

from src.output_paths import (
    ensure_output_path_within_root,
    render_output_relative_path,
)

log = logging.getLogger(__name__)
# use the compression that is default for parquet to ensure compatibility with other tools that read the generated parquet files. 
PARQUET_COMPRESSION = "snappy"


def _load_db_and_group_windows(db_path: str | Path):
    """Load a hoplite DB and index windows (offsets, and window ids) by source recording.

        Used by both:
        - `export_embeddings_table(...)`
        - `run_recognizers_over_db(...)`

        Purpose:
        - Share one canonical source-indexing step so export and classification
            start from the same DB view.

        Returns:
        - `db`: open `SQLiteUSearchDB` handle.
        - `data_by_source`: dict mapping source filename to a list of
            `(offset, window_id)` tuples.
    """
    db_path = Path(db_path)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path))

    recordings = {r.id: r.filename for r in db.get_all_recordings()}
    windows = db.get_all_windows()

    data_by_source: dict[str, list[tuple[float, int]]] = {}
    for window in windows:
        source = recordings[window.recording_id]
        offset = window.offsets[0]
        data_by_source.setdefault(source, []).append((offset, window.id))

    return db, data_by_source


def _load_recording_id_to_source(db_path: str | Path) -> dict[str, str]:
    """Load a hoplite DB and build recording_id -> source filename mapping."""
    db_path = Path(db_path)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path))
    recordings = db.get_all_recordings()
    # Normalize keys to strings because staged classify parquet may store IDs as
    # either integer or string depending on writer behavior.
    return {str(r.id): r.filename for r in recordings}


def _resolve_output_destination(
    output_template: str,
    source: str,
    analysis: str,
    output_path: Path,
    embeddings_table_format: str | None = None,
    filetype: str | None = None,
    recognizer_name: str | None = None,
    template_type: str = "embeddings",
) -> Path:
    """Render and validate output destination for one source item.

        Used by:
        - `export_embeddings_table(...)` to compute real export file destinations.
        - `run_recognizers_over_db(...)` to compute destination keys used for
            source grouping and final recognizer output file placement.

        Behavior:
        - Renders a relative path from `output_template` using source metadata,
            file extension, and table format token value.
        - Validates the rendered path is within `output_path`.

        Returns:
        - Absolute, validated destination path under `output_path`.
    """
    ext = f".{filetype}" if filetype is not None else ""
    rel_path = render_output_relative_path(
        template=output_template,
        audio_file=source,
        analysis=analysis,
        ext=ext,
        embeddings_table_format=embeddings_table_format,
        recognizer_name=recognizer_name,
        template_type=template_type,
    )
    return ensure_output_path_within_root(rel_path, output_path)


def export_classify_table(
    staging_path: str | Path,
    db_path: str | Path,
    output_path: str | Path,
    filetype: str,
    output_template: str,
    sourcemap: Callable[[str], str] | None = None,
    parquet_metadata: dict[str, str] | None = None,
    extra_columns: Callable[[str], dict[str, Any]] | None = None
):
    """Export staged base-model classify rows to templated output files."""
    if sourcemap is None:
        sourcemap = lambda x: x

    staging_path = Path(staging_path)
    if not staging_path.exists():
        log.warning("No staged classify rows found at %s - nothing to export.", staging_path)
        return

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(staging_path)
    if df.empty:
        log.info("Staged classify rows are empty - nothing to export.")
        staging_path.unlink(missing_ok=True)
        return

    required_cols = {"recording_id", "offset_s", "species", "score"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"Staged classify parquet is missing required columns: {missing_cols}"
        )

    recording_id_to_source = _load_recording_id_to_source(db_path)
    recording_ids = sorted(df["recording_id"].dropna().astype(str).unique().tolist())

    missing_recording_ids = [rid for rid in recording_ids if rid not in recording_id_to_source]
    if missing_recording_ids:
        raise ValueError(
            "Staged classify parquet contains recording_id values that are missing from the hoplite DB: "
            f"{missing_recording_ids}"
        )

    source_keys = sorted({recording_id_to_source[rid] for rid in recording_ids})
    data_by_source = {source: [] for source in source_keys}
    source_to_mapped_source = _build_and_validate_sourcemap_sources(data_by_source, sourcemap)
    extra_cols_by_source, extra_column_names = _resolve_extra_columns_by_source(source_keys, extra_columns)

    recording_id_to_mapped_source = {
        rid: source_to_mapped_source[recording_id_to_source[rid]]
        for rid in recording_ids
    }

    dest_paths_map = {
        source: _resolve_output_destination(
            output_template=output_template,
            source=source,
            analysis="classify_results",
            output_path=output_path,
            filetype=filetype,
            template_type="classify",
        )
        for source in source_keys
    }

    dest_paths_set = set(dest_paths_map.values())
    for dest in dest_paths_set:
        dest_inprogress = dest.with_suffix(dest.suffix + ".inprogress")
        if dest_inprogress.exists():
            dest_inprogress.unlink()
            log.info("Removed stale in-progress file %s", dest_inprogress)

    inprogress_paths: set[Path] = set()
    parquet_writers: dict[Path, pq.ParquetWriter] = {}

    try:
        for i, recording_id in enumerate(recording_ids, 1):
            source = recording_id_to_source[recording_id]
            source_rows = df[df["recording_id"].astype(str) == recording_id].copy()
            if source_rows.empty:
                continue

            source_rows = source_rows.rename(columns={"offset_s": "offset", "species": "label"})
            source_rows["source"] = recording_id_to_mapped_source[recording_id]
            source_rows["channel"] = 0
            source_rows = source_rows[["source", "channel", "offset", "label", "score"]]
            source_rows = _apply_extra_columns(
                source_rows,
                extra_cols_by_source[source],
                extra_column_names,
            )

            dest = dest_paths_map[source]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest_inprogress = dest.with_suffix(dest.suffix + ".inprogress")

            if filetype == "parquet":
                write_inprogress_parquet(dest_inprogress, source_rows, parquet_writers)
            elif filetype == "csv":
                write_inprogress_csv(dest_inprogress, source_rows)
            else:
                raise ValueError(f"Unsupported filetype: {filetype}")

            inprogress_paths.add(dest_inprogress)
            log.info(
                "  [%d/%d] Wrote classify rows for %s (%d row(s))",
                i,
                len(recording_ids),
                source,
                len(source_rows),
            )
    finally:
        for writer in parquet_writers.values():
            writer.close()

    for inprogress_path in inprogress_paths:
        finalize_inprogress_file(
            inprogress_path,
            filetype,
            parquet_metadata=parquet_metadata,
        )

    staging_path.unlink(missing_ok=True)


def _build_and_validate_sourcemap_sources(
    data_by_source: dict[str, list[tuple[float, int]]],
    sourcemap,
) -> dict[str, str]:
    """Build source->mapped_source and ensure sourcemap values are unique.

    Raises:
        ValueError: if two different input sources map to the same sourcemap
            source value.
    """
    source_to_mapped_source = {
        source: sourcemap(source)
        for source in data_by_source
    }

    mapped_values = list(source_to_mapped_source.values())
    if len(set(mapped_values)) != len(mapped_values):
        reverse_map: dict[str, list[str]] = {}
        for source, mapped_source in source_to_mapped_source.items():
            reverse_map.setdefault(mapped_source, []).append(source)

        collisions = {
            mapped_source: sorted(sources)
            for mapped_source, sources in reverse_map.items()
            if len(sources) > 1
        }
        raise ValueError(
            "Sourcemap collision: multiple input sources map to the same output source value. "
            f"Collisions: {collisions}"
        )

    return source_to_mapped_source


def _resolve_extra_columns_by_source(
    sources: list[str],
    extra_columns: Callable[[str], dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Resolve extra column values for each source and collect all column names."""
    if extra_columns is None:
        return {source: {} for source in sources}, []

    values_by_source: dict[str, dict[str, Any]] = {}
    all_extra_cols: set[str] = set()

    for source in sources:
        values = extra_columns(source)
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError("extra_columns mapper must return a dictionary")
        values_by_source[source] = values
        all_extra_cols.update(values.keys())

    return values_by_source, sorted(all_extra_cols)


def _apply_extra_columns(
    df: pd.DataFrame,
    extra_values: dict[str, Any],
    extra_column_names: list[str],
) -> pd.DataFrame:
    """Append extra columns to a DataFrame with stable column presence/order."""
    if not extra_column_names:
        return df

    out = df.copy()
    collisions = [name for name in extra_column_names if name in out.columns]
    if collisions:
        raise ValueError(f"extra_columns contain reserved/existing column names: {collisions}")

    for col in extra_column_names:
        out[col] = extra_values.get(col)

    return out


def export_embeddings_table(
    db_path: str | Path,
    output_path: str | Path,
    table_format: str,
    filetype: str,
    output_template: str,
    sourcemap=None,
    parquet_metadata: dict[str, str] | None = None,
):
    """Export embeddings from a perch-hoplite database to tabular files.

    Produces one file per unique source (recording filename), per format.
    Supports both parquet and CSV formats with serialized or column-based embeddings.

    Args:
        db_path: Path to the hoplite database directory.
        output_path: Directory to write files to.
        table_format: Either "serialized" or "columns", determines how embeddings are stored in the output table.
        filetype: Either "parquet" or "csv", determines the output file format.
        output_template: Relative path template that determines the
            destination. 
        sourcemap: Optional function mapping a source filename to an output
            source value written into the rows. If None, the recording
            filename is used unchanged.
        parquet_metadata: Optional metadata key/value pairs to write into
            parquet file footer metadata.
    """

    if sourcemap is None:
        sourcemap = lambda x: x


    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    db, data_by_source = _load_db_and_group_windows(db_path)
    if not data_by_source:
        log.warning("No embeddings found in database - nothing to export.")
        return

    source_to_mapped_source = _build_and_validate_sourcemap_sources(data_by_source, sourcemap)

    total_refs = sum(len(entries) for entries in data_by_source.values())
    log.info("Preparing %d embedding reference(s) from database...", total_refs)

    log.info("Exporting %d source(s) to %s %s format...",
             len(data_by_source), filetype, table_format)


    # Precompute all destinations so we can clear stale in-progress files
    # before any new writes begin.

    # map sources to destinations for this format (each format can have a different destination due to {embeddings_table_format} token and file extension)
    dest_paths_map = {
        source: _resolve_output_destination(
            output_template,
            source=source,
            analysis="embeddings",
            output_path=output_path,
            embeddings_table_format=table_format,
            filetype=filetype,
            template_type="embeddings",
        )
        for source in data_by_source
    }

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
    try:
        for i, (source, entries) in enumerate(data_by_source.items(), 1):
            entries.sort(key=lambda x: x[0])
            output_source_value = source_to_mapped_source[source]

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
    finally:
        for writer in parquet_writers.values():
            writer.close()

    for inprogress_path in inprogress_paths:
        finalize_inprogress_file(
            inprogress_path,
            filetype,
            parquet_metadata=parquet_metadata,
        )


def run_recognizers_over_db(
    db_path: str | Path,
    output_parent: str | Path,
    recognizers,
    recognizer_results_filetype: str,
    sourcemap: Callable[[str], str],
    output_template,
    parquet_metadata: dict[str, str] | None = None,
    extra_columns: Callable[[str], dict] | None = None
):
    """Run embeddings-classifier over DB embeddings, writing results per source.

    Called by:
    - `src.embed.embed(...)` when config contains non-empty `recognizers`.

    Processes one source recording at a time. For each source, builds a
    feature-column DataFrame, calls `classify_table` once, and appends any
    result rows to per-classifier `.inprogress` output files. All in-progress
    files are finalized (deduped, sorted, renamed) after all sources complete.

    Args:
        db_path: hoplite DB directory.
        output_parent: output root directory.
        recognizers: Normalized `ClassifierConfigList` for
            `embeddings-classifier`.
        recognizer_filetype: Output extension (`csv` or `parquet`).
        sourcemap: optional source-name remapper.
        output_template: optional path template; defaults to runner default.
    """
    from embeddings_classifier import classify_table
    if sourcemap is None:
        sourcemap = lambda x: x

    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)

    db, data_by_source = _load_db_and_group_windows(db_path)
    if not data_by_source:
        log.warning("No embeddings found in database - nothing to classify.")
        return

    source_to_mapped_source = _build_and_validate_sourcemap_sources(data_by_source, sourcemap)
    source_keys = sorted(data_by_source.keys())
    extra_cols_by_source, extra_column_names = _resolve_extra_columns_by_source(source_keys, extra_columns)

    # cfg is a ClassifierConfigList which uses "classifer_name" to refer to what we call "recognizer_name" for templating purposes. 
    recognizer_names = [cfg.classifier_name for cfg in recognizers.configs]

    dest_paths_map = {
        source: {
            recognizer_name: _resolve_output_destination(
                output_template,
                source=source,
                analysis="recognizer_results",
                output_path=output_parent,
                filetype=recognizer_results_filetype,
                recognizer_name=recognizer_name,
                template_type="recognizer",
            )
            for recognizer_name in recognizer_names}
        for source in data_by_source
    }

    # get a set of unique output paths so we can clear stale in-progress files before processing starts
    dest_paths_set = set()
    for source, recognizer_map in dest_paths_map.items():
        for recognizer_name, path in recognizer_map.items():
            dest_paths_set.add(path)

    log.info(
        "Running recognizer(s) over %d source(s) into %d output group(s)...",
        len(data_by_source),
        len(dest_paths_set),
    )

    parquet_writers: dict[Path, pq.ParquetWriter] = {}
    inprogress_paths: set[Path] = set()

    # Clear stale in-progress outputs before processing starts.
    for recognizer_output_dest in dest_paths_set:
        dest_inprogress = recognizer_output_dest.with_suffix(recognizer_output_dest.suffix + ".inprogress")
        if dest_inprogress.exists():
            dest_inprogress.unlink()
            log.info("Removed stale in-progress file %s", dest_inprogress)

    try:
        for i, (source, entries) in enumerate(data_by_source.items(), 1):
            entries.sort(key=lambda x: x[0])
            output_source_value = source_to_mapped_source[source]
            recognizer_output_dest = dest_paths_map[source]
            source_df = build_rows(output_source_value, entries, "columns", db)

            results_written = 0
            table = pa.Table.from_pandas(source_df, preserve_index=False)

            # output path is None because we don't want embeddings-classifier to write output.
            # We handle the writing of output here in this app
            results = classify_table(table, recognizers, output_path=None)

            failed = [item for item in results if not item.success]
            if failed:
                errors = "; ".join(
                    item.error or item.message or f"unknown failure ({item.config.classifier_name})"
                    for item in failed
                )
                raise RuntimeError(f"Classification failed for {source}: {errors}")

            # one item for each recognizer
            for item in results:
                result_table = item.result_table
                if result_table is None:
                    continue

                recognizer_name = item.config.classifier_name
                recognizer_output_file = recognizer_output_dest[recognizer_name]
                dest_inprogress = recognizer_output_file.with_suffix(recognizer_output_file.suffix + ".inprogress")

                recognizer_output_file.parent.mkdir(parents=True, exist_ok=True)

                result_df = result_table.to_pandas()
                result_df = _apply_extra_columns(
                    result_df,
                    extra_cols_by_source[source],
                    extra_column_names,
                )
                if recognizer_results_filetype == "parquet":
                    write_inprogress_parquet(dest_inprogress, result_df, parquet_writers)
                else:
                    write_inprogress_csv(dest_inprogress, result_df)

                inprogress_paths.add(dest_inprogress)
                results_written += len(result_df)

            log.info(
                "  [%d/%d] Classified %s (%d input rows, %d output row(s))",
                i,
                len(data_by_source),
                source,
                len(source_df),
                results_written,
            )
    finally:
        for writer in parquet_writers.values():
            writer.close()

    for inprogress_path in inprogress_paths:
        finalize_inprogress_file(inprogress_path, recognizer_results_filetype, parquet_metadata=parquet_metadata)



def finalize_inprogress_file(
    inprogress_path: Path,
    filetype: str,
    parquet_metadata: dict[str, str] | None = None,
) -> None:
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
            table = pa.Table.from_pandas(df, preserve_index=False)

            schema_metadata = dict(table.schema.metadata or {})
            if parquet_metadata:
                schema_metadata.update({
                    str(key).encode("utf-8"): str(value).encode("utf-8")
                    for key, value in parquet_metadata.items()
                })
            table = table.replace_schema_metadata(schema_metadata)

            pq.write_table(table, final_path, compression=PARQUET_COMPRESSION)
        elif filetype == 'csv':
            df.to_csv(final_path, index=False)

        inprogress_path.unlink()
        log.info("Finalized %s (deduped, sorted, renamed)", final_path)
    except (OSError, ValueError) as e:
        log.error("Failed to finalize %s: %s", inprogress_path, e)
        raise


def build_rows(source_value, entries: list[tuple[float, int]], embeddings_table_format, db) -> pd.DataFrame:
    """Build a DataFrame for one source by fetching embeddings by window id."""
    if embeddings_table_format == 'columns':
        embeddings = [np.asarray(db.get_embedding(embedding_id), dtype=np.float32)
                      for _, embedding_id in entries]
        embeddings_matrix = np.vstack(embeddings)
        col_names = data_frames.embedding_col_names(embeddings_matrix.shape[1])

        df = pd.DataFrame(embeddings_matrix, columns=col_names)
        offsets = [offset for offset, _ in entries]
        df.insert(0, 'offset', offsets)
        df.insert(0, 'channel', 0)
        df.insert(0, 'source', source_value)
        return df

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
        