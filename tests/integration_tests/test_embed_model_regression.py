"""Regression tests for model-specific embed/export outputs."""

import shutil

import numpy as np
import pandas as pd

from src import embed
from src.config import EmbeddingsFormat

from .embed_helpers import FIXTURES_DIR


AUDIO_4731099 = "20241222T110000+0700_Site-3_4731099.wav"
EXPECTED_PARQUET_4731099 = FIXTURES_DIR / "embeddings" / "4731099.embeddings.parquet"


def test_perch8_columns_regression(workspace):
    """Compare perch_8 columns export against a known-good fixture parquet."""
    source, output = workspace
    site = source / "site1"
    site.mkdir()
    shutil.copy(FIXTURES_DIR / "audio" / AUDIO_4731099, site)

    config = {
        "source": str(source),
        "output": str(output),
        "db_path": str(output / "db"),
        "model_choice": "perch_8",
        "dataset_name": "search_set",
    }
    embed.create_database(config)
    embed.export_embeddings_table(
        db_path=str(output / "db"),
        output_path=str(output / "embeddings"),
        embeddings_formats=[EmbeddingsFormat("parquet", "columns")],
    )

    parquet_files = list((output / "embeddings").rglob("*.parquet"))
    assert len(parquet_files) == 1, f"Expected 1 parquet file, found {len(parquet_files)}"

    actual = pd.read_parquet(parquet_files[0])
    expected = pd.read_parquet(EXPECTED_PARQUET_4731099)

    assert len(actual) == len(expected), (
        f"Row count mismatch: actual={len(actual)}, expected={len(expected)}"
    )
    assert set(actual.columns) == set(expected.columns), (
        f"Column mismatch: extra={set(actual.columns) - set(expected.columns)}, "
        f"missing={set(expected.columns) - set(actual.columns)}"
    )
    assert list(actual.columns) == list(expected.columns), (
        f"Column order differs: actual starts {list(actual.columns[:5])}, "
        f"expected starts {list(expected.columns[:5])}"
    )

    np.testing.assert_array_equal(
        actual["offset"].values.astype(float),
        expected["offset"].values.astype(float),
    )
    np.testing.assert_array_equal(actual["channel"].values, expected["channel"].values)

    feature_cols = [c for c in expected.columns if c.startswith("f")]
    assert len(feature_cols) == 1280

    actual_features = actual[feature_cols].values.astype(np.float16)
    expected_features = expected[feature_cols].values.astype(np.float16)

    np.testing.assert_allclose(
        actual_features,
        expected_features,
        atol=np.finfo(np.float16).eps,
        rtol=0,
    )
