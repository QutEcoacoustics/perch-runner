"""Integration and unit-style tests for export_embeddings_table."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from perch_hoplite.db import sqlite_usearch_impl

from src import data_frames
from src.db_to_table import export_embeddings_table
from src.version import MODELS

from .embed_helpers import A2O_FLAC, FIXTURE_DBS


NESTED_TEMPLATE = "{parents}/{basename}/{analysis}{ext}"


def test_empty_db_logs_warning_returns_early(workspace, caplog):
    source, output = workspace
    db_path = source / "hoplite"
    db_path.mkdir()
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(1536)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path), usearch_cfg)
    assert db.count_embeddings() == 0

    import logging

    with caplog.at_level(logging.WARNING, logger="src.db_to_table"):
        export_embeddings_table(
            db_path=str(db_path),
            output_path=str(output),
            table_format="serialized",
            filetype="parquet",
            output_template=NESTED_TEMPLATE,
        )

    assert any("No embeddings found" in r.message for r in caplog.records)


def test_custom_sourcemap_function(workspace):
    _, output = workspace

    def custom_map(source_filename):
        return f"custom::{source_filename}"

    export_embeddings_table(
        db_path="tests/files/hoplite_perch_v2",
        output_path=str(output),
        table_format="serialized",
        filetype="parquet",
        output_template=NESTED_TEMPLATE,
        sourcemap=custom_map,
    )

    parquets = sorted(output.rglob("*.parquet"))
    assert parquets
    df = pd.read_parquet(parquets[0])
    assert df["source"].str.startswith("custom::").all()


def test_parquet_metadata_written(workspace):
    _, output = workspace

    metadata = {
        "perch_runner.version": "test-version",
        "perch_hoplite.version": "test-hoplite",
        "perch_runner.config_json": '{"embed": true}',
    }

    export_embeddings_table(
        db_path="tests/files/hoplite_perch_v2",
        output_path=str(output),
        table_format="serialized",
        filetype="parquet",
        output_template=NESTED_TEMPLATE,
        parquet_metadata=metadata,
    )

    parquets = sorted(output.rglob("*.parquet"))
    assert parquets

    footer_metadata = pq.read_metadata(parquets[0]).metadata or {}
    assert footer_metadata[b"perch_runner.version"] == b"test-version"
    assert footer_metadata[b"perch_hoplite.version"] == b"test-hoplite"


def test_finalize_failure_raises(workspace):
    _, output = workspace

    with mock.patch("src.db_to_table.pq.write_table", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            export_embeddings_table(
                db_path="tests/files/hoplite_perch_v2",
                output_path=str(output),
                table_format="serialized",
                filetype="parquet",
                output_template=NESTED_TEMPLATE,
            )


def test_export_serialized_fixture_db(tmp_path):
    output = tmp_path / "embeddings"

    export_embeddings_table(
        db_path=str(FIXTURE_DBS["perch_v2"]),
        output_path=str(output),
        table_format="serialized",
        filetype="parquet",
        output_template=NESTED_TEMPLATE,
    )

    parquets = list(output.rglob("embeddings.parquet"))
    assert parquets

    df = pd.read_parquet(parquets[0])
    emb = data_frames.deserialize_array(df["embeddings"].iloc[0], dtype=np.float32)
    assert len(emb) == 1536
    assert df["offset"].is_monotonic_increasing


@pytest.mark.parametrize("model_choice", MODELS.keys(), ids=MODELS.keys())
def test_export_columns_all_models(model_choice, tmp_path):
    output = tmp_path / "embeddings"
    expected_dim = MODELS[model_choice]["embedding_dim"]

    export_embeddings_table(
        db_path=str(FIXTURE_DBS[model_choice]),
        output_path=str(output),
        table_format="columns",
        filetype="parquet",
        output_template=NESTED_TEMPLATE,
    )

    parquets = list(output.rglob("embeddings.parquet"))
    assert parquets

    df = pd.read_parquet(parquets[0])
    feature_cols = [c for c in df.columns if c.startswith("f")]
    assert len(feature_cols) == expected_dim


def test_export_both_formats_by_two_calls(tmp_path):
    output = tmp_path / "embeddings"
    template = "{embeddings_table_format}/{parents}/{basename}/{analysis}{ext}"

    export_embeddings_table(
        db_path="tests/files/hoplite_perch_v2",
        output_path=str(output),
        table_format="serialized",
        filetype="parquet",
        output_template=template,
    )
    export_embeddings_table(
        db_path="tests/files/hoplite_perch_v2",
        output_path=str(output),
        table_format="columns",
        filetype="parquet",
        output_template=template,
    )

    ser_path = output / "serialized" / "one" / "100sec.wav" / "embeddings.parquet"
    col_path = output / "columns" / "one" / "100sec.wav" / "embeddings.parquet"
    assert ser_path.exists()
    assert col_path.exists()


def test_a2o_flac_export_parquet(tmp_path):
    output = tmp_path / "embeddings"

    export_embeddings_table(
        db_path="tests/files/hoplite_a2o",
        output_path=str(output),
        table_format="serialized",
        filetype="parquet",
        output_template=NESTED_TEMPLATE,
    )

    parquet_path = output / f"Minjerribah-Dry-B/{A2O_FLAC}" / "embeddings.parquet"
    assert parquet_path.exists()
