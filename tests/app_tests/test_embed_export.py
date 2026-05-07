"""
Parquet export tests (pre-generated fixture DBs — no model loaded).

These tests use databases that were generated offline and committed
to tests/files/. No TensorFlow or CNN inference is needed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import embed
from src import data_frames

from .embed_helpers import FIXTURES_DIR, A2O_FLAC, MODELS_TO_CACHE, MODEL_IDS, FIXTURE_DBS


def test_export_as_parquet(tmp_path):
    """Export the perch_v2 fixture DB to serialized parquet."""
    output = tmp_path / "embeddings"

    embed.export_as_parquet(
        db_path="tests/files/hoplite",
        output_path=str(output),
    )

    assert (output / "one" / "100sec.wav" / "embeddings.parquet").exists()
    assert (output / "two" / "100sec.wav" / "embeddings.parquet").exists()

    df = pd.read_parquet(output / "one" / "100sec.wav" / "embeddings.parquet")
    assert len(df) > 0
    assert "source" in df.columns
    assert "channel" in df.columns
    assert "offset" in df.columns
    assert "embeddings" in df.columns
    emb = data_frames.deserialize_array(df["embeddings"].iloc[0])
    assert len(emb) > 0
    assert emb.dtype == np.float32
    assert df["source"].iloc[0] == "one/100sec.wav"
    assert df["offset"].is_monotonic_increasing


def test_export_as_parquet_columns(tmp_path):
    """Export in columns format (f0000, f0001, ...)."""
    output = tmp_path / "embeddings"

    embed.export_as_parquet(
        db_path="tests/files/hoplite",
        output_path=str(output),
        as_serialized=False,
        as_columns=True,
    )

    assert (output / "one" / "100sec.wav" / "embeddings.parquet").exists()

    df = pd.read_parquet(output / "one" / "100sec.wav" / "embeddings.parquet")
    assert len(df) > 0
    assert "f0000" in df.columns
    assert "f0001" in df.columns
    assert "embeddings" not in df.columns
    assert df["offset"].is_monotonic_increasing


def test_export_as_parquet_both(tmp_path):
    """Export both serialized and columns formats simultaneously."""
    output = tmp_path / "embeddings"

    embed.export_as_parquet(
        db_path="tests/files/hoplite",
        output_path=str(output),
        as_serialized=True,
        as_columns=True,
    )

    assert (output / "serialized" / "one" / "100sec.wav" / "embeddings.parquet").exists()
    assert (output / "columns" / "one" / "100sec.wav" / "embeddings.parquet").exists()

    df_ser = pd.read_parquet(output / "serialized" / "one" / "100sec.wav" / "embeddings.parquet")
    assert "embeddings" in df_ser.columns
    assert "f0000" not in df_ser.columns

    df_col = pd.read_parquet(output / "columns" / "one" / "100sec.wav" / "embeddings.parquet")
    assert "f0000" in df_col.columns
    assert "embeddings" not in df_col.columns

    assert len(df_ser) == len(df_col)

    first_serialized = data_frames.deserialize_array(df_ser["embeddings"].iloc[0])
    feature_cols = [c for c in df_col.columns if c.startswith("f")]
    first_columns = df_col[feature_cols].iloc[0].values.astype(np.float32)
    np.testing.assert_array_almost_equal(first_serialized, first_columns)


def test_a2o_flac_export_parquet(tmp_path):
    """
    Export the A2O fixture DB: verifies parquet path mirrors source structure.
    """
    output = tmp_path / "embeddings"

    embed.export_as_parquet(
        db_path="tests/files/hoplite_a2o",
        output_path=str(output),
    )

    parquet_path = output / f"Minjerribah-Dry-B/{A2O_FLAC}" / "embeddings.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 6
    assert df.shape[1] == 4
    assert "embeddings" in df.columns
    assert df["source"].iloc[0] == f"Minjerribah-Dry-B/{A2O_FLAC}"
    assert df["offset"].is_monotonic_increasing


@pytest.mark.parametrize("model_choice,expected_dim", MODELS_TO_CACHE, ids=MODEL_IDS)
def test_export_parquet_model(model_choice, expected_dim, tmp_path):
    """
    Export each model's fixture DB and verify embedding dimensionality.
    No CNN — uses pre-generated databases.
    """
    output = tmp_path / "embeddings"
    db_path = FIXTURE_DBS[model_choice]

    embed.export_as_parquet(
        db_path=str(db_path),
        output_path=str(output),
    )

    parquets = list(output.rglob("embeddings.parquet"))
    assert len(parquets) > 0

    df = pd.read_parquet(parquets[0])
    assert len(df) > 0
    assert "embeddings" in df.columns
    assert df["offset"].is_monotonic_increasing

    emb = data_frames.deserialize_array(df["embeddings"].iloc[0])
    assert len(emb) == expected_dim
