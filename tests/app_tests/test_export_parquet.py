"""
Tests for export_as_parquet function.

Includes both unit tests (with mocked DBs) and integration tests (with real fixture DBs).
"""
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src import embed
from src import data_frames
from src.config import EmbeddingsFormat
from perch_hoplite.db import sqlite_usearch_impl

from .embed_helpers import FIXTURES_DIR, A2O_FLAC, FIXTURE_DBS
from src.version import MODELS


# ---------------------------------------------------------------------------
# Unit Tests: export_as_parquet with mocked databases
# ---------------------------------------------------------------------------

class TestExportAsParquetUnit:
    """Unit tests for export_as_parquet with mocked/temporary databases."""

    def test_empty_db_logs_warning_returns_early(self, workspace, caplog):
        """Empty DB logs warning and returns without writing files."""
        source, output = workspace
        db_path = source / "hoplite"
        db_path.mkdir()
        usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(1536)
        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path), usearch_cfg)
        assert db.count_embeddings() == 0

        import logging
        with caplog.at_level(logging.WARNING, logger="src.embed_export_table"):
            embed.export_embeddings_table(
                db_path=str(db_path),
                output_path=str(output),
                embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
            )

        assert any("No embeddings found" in r.message for r in caplog.records)
        assert not list(output.rglob("*.parquet")) if output.exists() else True

    def test_creates_output_dir_if_missing(self, workspace):
        """Output directory is created if it doesn't exist."""
        source, output = workspace
        output = output / "deeply" / "nested" / "output"
        assert not output.exists()

        embed.export_embeddings_table(
            db_path="tests/files/hoplite_perch_v2",
            output_path=str(output),
            embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
        )

        assert output.exists()
        assert len(list(output.rglob("*.parquet"))) > 0

    def test_custom_sourcemap_function(self, workspace):
        """Custom sourcemap changes output paths."""
        source, output = workspace
        output = output / "embeddings"

        def custom_map(source_filename):
            return f"custom::{source_filename}"

        embed.export_embeddings_table(
            db_path="tests/files/hoplite_perch_v2",
            output_path=str(output),
            embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
            sourcemap=custom_map,
        )

        parquets = sorted(output.rglob("*.parquet"))
        assert len(parquets) > 0
        df = pd.read_parquet(parquets[0])
        assert df["source"].str.startswith("custom::").all()

    def test_multiple_sources_in_db(self, workspace):
        """Pre-generated DB with 2 sources produces 2 parquet file trees."""
        source, output = workspace
        output = output / "embeddings"

        embed.export_embeddings_table(
            db_path="tests/files/hoplite_perch_v2",
            output_path=str(output),
            embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
        )

        parquets = sorted(output.rglob("*.parquet"))
        assert len(parquets) == 2

    def test_offsets_sorted_per_source(self, workspace):
        """Parquet output has offsets sorted within each source."""
        source, output = workspace
        output = output / "embeddings"

        embed.export_embeddings_table(
            db_path="tests/files/hoplite_perch_v2",
            output_path=str(output),
            embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
        )

        for pq in output.rglob("*.parquet"):
            df = pd.read_parquet(pq)
            assert df["offset"].is_monotonic_increasing, f"Offsets not sorted in {pq}"


# ---------------------------------------------------------------------------
# Integration Tests: export_as_parquet with real fixture databases
# ---------------------------------------------------------------------------

def test_export_as_parquet_serialized(tmp_path):
    """Export fixture DB to serialized parquet for each model."""
    output = tmp_path / "embeddings"
    db_path = FIXTURE_DBS["perch_v2"]

    embed.export_embeddings_table(
        db_path=str(db_path),
        output_path=str(output),
        embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
    )

    parquets = list(output.rglob("embeddings.parquet"))
    assert len(parquets) > 0

    df = pd.read_parquet(parquets[0])
    assert len(df) > 0
    assert "embeddings" in df.columns
    assert "source" in df.columns
    assert "offset" in df.columns
    assert df["offset"].is_monotonic_increasing

    emb = data_frames.deserialize_array(df["embeddings"].iloc[0], dtype=np.float32)
    assert len(emb) == 1536
    assert emb.dtype == np.float32


@pytest.mark.parametrize("model_choice", MODELS.keys(), ids=MODELS.keys())
def test_export_as_parquet_all_models(model_choice, tmp_path):
    """Export each model's fixture DB and verify embedding dimensionality (serialized)."""
    expected_dim = MODELS[model_choice]["embedding_dim"]
    output = tmp_path / "embeddings"
    db_path = FIXTURE_DBS[model_choice]

    embed.export_embeddings_table(
        db_path=str(db_path),
        output_path=str(output),
        embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
    )

    parquets = list(output.rglob("embeddings.parquet"))
    assert len(parquets) > 0

    df = pd.read_parquet(parquets[0])
    assert len(df) > 0
    assert "embeddings" in df.columns
    assert df["offset"].is_monotonic_increasing

    emb = data_frames.deserialize_array(df["embeddings"].iloc[0], dtype=np.float32)
    assert len(emb) == expected_dim
    assert emb.dtype == np.float32


@pytest.mark.parametrize("model_choice", MODELS.keys(), ids=MODELS.keys())
def test_export_as_parquet_columns_all_models(model_choice, tmp_path):
    """Export each model's fixture DB in columns format (f0000, f0001, ...)."""
    expected_dim = MODELS[model_choice]["embedding_dim"]
    output = tmp_path / "embeddings"
    db_path = FIXTURE_DBS[model_choice]

    embed.export_embeddings_table(
        db_path=str(db_path),
        output_path=str(output),
        embeddings_formats=[EmbeddingsFormat("parquet", "columns")],
    )

    parquets = list(output.rglob("embeddings.parquet"))
    assert len(parquets) > 0

    df = pd.read_parquet(parquets[0])
    assert len(df) > 0
    assert "f0000" in df.columns
    assert "f0001" in df.columns
    assert "embeddings" not in df.columns
    assert df["offset"].is_monotonic_increasing

    feature_cols = [c for c in df.columns if c.startswith("f")]
    assert len(feature_cols) == expected_dim


def test_export_as_parquet_both_formats(tmp_path):
    """Export both serialized and columns formats simultaneously."""
    output = tmp_path / "embeddings"

    embed.export_embeddings_table(
        db_path="tests/files/hoplite_perch_v2",
        output_path=str(output),
        embeddings_formats=[
            EmbeddingsFormat("parquet", "serialized"),
            EmbeddingsFormat("parquet", "columns"),
        ],
        output_template="{embedding_table_format}/{parents}/{basename}{ext}/embeddings",
    )

    assert (output / "serialized" / "one" / "100sec.wav.parquet" / "embeddings.parquet").exists()
    assert (output / "columns" / "one" / "100sec.wav.parquet" / "embeddings.parquet").exists()

    df_ser = pd.read_parquet(output / "serialized" / "one" / "100sec.wav.parquet" / "embeddings.parquet")
    assert "embeddings" in df_ser.columns
    assert "f0000" not in df_ser.columns

    df_col = pd.read_parquet(output / "columns" / "one" / "100sec.wav.parquet" / "embeddings.parquet")
    assert "f0000" in df_col.columns
    assert "embeddings" not in df_col.columns

    assert len(df_ser) == len(df_col)

    first_serialized = data_frames.deserialize_array(df_ser["embeddings"].iloc[0], dtype=np.float32)
    feature_cols = [c for c in df_col.columns if c.startswith("f")]
    first_columns = df_col[feature_cols].iloc[0].values.astype(np.float32)
    np.testing.assert_array_almost_equal(first_serialized, first_columns)


def test_a2o_flac_export_parquet(tmp_path):
    """
    Export the A2O fixture DB: verifies parquet path mirrors source structure.
    """
    output = tmp_path / "embeddings"

    embed.export_embeddings_table(
        db_path="tests/files/hoplite_a2o",
        output_path=str(output),
        embeddings_formats=[EmbeddingsFormat("parquet", "serialized")],
    )

    parquet_path = output / "Minjerribah-Dry-B/20220502T075930+1000_Minjerribah-Dry-B_1088507.flac" / "embeddings.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 6
    assert df.shape[1] == 4
    assert "embeddings" in df.columns
    assert df["source"].iloc[0] == f"Minjerribah-Dry-B/{A2O_FLAC}"
    assert df["offset"].is_monotonic_increasing
