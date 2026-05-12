"""
Full pipeline and model tests (real CNN — loads TensorFlow).

These tests actually run inference through the embedding model.
Each test is marked forked to run in a subprocess and prevent OOM.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import embed
from src import data_frames
from src.version import MODELS
from perch_hoplite.db import sqlite_usearch_impl

from .embed_helpers import FIXTURES_DIR

AUDIO_4731099 = "20241222T110000+0700_Site-3_4731099.wav"
EXPECTED_PARQUET_4731099 = FIXTURES_DIR / "embeddings" / "4731099.embeddings.parquet"


@pytest.fixture
def workspace(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    return source, output


# ---------------------------------------------------------------------------
# Full pipeline (embed + export, one model)
# ---------------------------------------------------------------------------

def test_full_pipeline(workspace):
    """
    End-to-end: embed audio from a nested folder, export to parquet,
    verify output shape, columns, and source values.
    """
    source, output = workspace
    one = source / "one"
    one.mkdir()
    shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", one)

    config = {"source": str(source), "output": str(output), "model_choice": "perch_v2", "dataset_name": "search_set"}
    embed.create_database(config)
    embed.export_as_parquet(
        db_path=str(output / "hoplite"),
        output_path=str(output / "embeddings"),
    )

    parquet_path = output / "embeddings" / "one" / "100sec.wav" / "embeddings.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 19
    assert df.shape[1] == 4
    assert "embeddings" in df.columns
    emb = data_frames.deserialize_array(df["embeddings"].iloc[0], dtype=np.float16)
    assert len(emb) == 1536
    assert df["source"].iloc[0] == "one/100sec.wav"
    assert df["offset"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Model validation (each model loads, embeds, correct dimensionality)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_choice", MODELS.keys(), ids=MODELS.keys())
def test_create_db_model(model_choice, workspace):
    """
    Creates embeddings with the given model. Verifies correct dimensionality.
    Models must be pre-cached (downloaded during Docker build).
    """
    expected_dim = MODELS[model_choice]["embedding_dim"]
    source, output = workspace
    site = source / "site1"
    site.mkdir()
    shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

    config = {"source": str(source), "output": str(output), "model_choice": model_choice, "dataset_name": "search_set"}
    embed.create_database(config)

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "hoplite"))
    assert db.count_embeddings() > 0
    assert len(db.get_all_recordings()) == 1

    windows = db.get_all_windows()
    emb = db.get_embedding(windows[0].id)
    assert emb.shape == (expected_dim,)


# ---------------------------------------------------------------------------
# Regression: perch_8 + columns format against known-good fixture
# ---------------------------------------------------------------------------

def test_perch8_columns_regression(workspace):
    """
    Embed 4731099.wav with perch_8, export as columns,
    compare output against known-good fixture parquet.
    This existing embedding was produced by the old perch-runner. This test
    serves as a regression check to ensure the new embedding pipeline produces the equivalent output.
    """
    source, output = workspace
    site = source / "site1"
    site.mkdir()
    shutil.copy(FIXTURES_DIR / "audio" / AUDIO_4731099, site)

    config = {
        "source": str(source),
        "output": str(output),
        "model_choice": "perch_8",
        "dataset_name": "search_set",
    }
    embed.create_database(config)
    embed.export_as_parquet(
        db_path=str(output / "hoplite"),
        output_path=str(output / "embeddings"),
        as_serialized=False,
        as_columns=True,
    )

    parquet_files = list((output / "embeddings").rglob("*.parquet"))
    assert len(parquet_files) == 1, f"Expected 1 parquet file, found {len(parquet_files)}"

    actual = pd.read_parquet(parquet_files[0])
    expected = pd.read_parquet(EXPECTED_PARQUET_4731099)

    # Same row count
    assert len(actual) == len(expected), (
        f"Row count mismatch: actual={len(actual)}, expected={len(expected)}"
    )

    # Same column set (order-insensitive)
    assert set(actual.columns) == set(expected.columns), (
        f"Column mismatch: extra={set(actual.columns) - set(expected.columns)}, "
        f"missing={set(expected.columns) - set(actual.columns)}"
    )

    # Same column order
    assert list(actual.columns) == list(expected.columns), (
        f"Column order differs: actual starts {list(actual.columns[:5])}, "
        f"expected starts {list(expected.columns[:5])}"
    )

    # Same offsets
    np.testing.assert_array_equal(
        actual["offset"].values.astype(float),
        expected["offset"].values.astype(float),
    )

    # Same channel values
    np.testing.assert_array_equal(actual["channel"].values, expected["channel"].values)

    # Compare embedding values — expected is float16, actual is float64.
    # Allow tolerance of 1 float16 ULP (≈0.00049) for rounding differences.
    feature_cols = [c for c in expected.columns if c.startswith("f")]
    assert len(feature_cols) == 1280

    actual_features = actual[feature_cols].values.astype(np.float16)
    expected_features = expected[feature_cols].values.astype(np.float16)

    np.testing.assert_allclose(
        actual_features, expected_features,
        atol=np.finfo(np.float16).eps,
        rtol=0,
    )
