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
from perch_hoplite.db import sqlite_usearch_impl

from .embed_helpers import FIXTURES_DIR, MODELS_TO_CACHE, MODEL_IDS


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
    emb = data_frames.deserialize_array(df["embeddings"].iloc[0])
    assert len(emb) == 1536
    assert df["source"].iloc[0] == "one/100sec.wav"
    assert df["offset"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Model validation (each model loads, embeds, correct dimensionality)
# These are the only tests allowed to download models (used during build).
# ---------------------------------------------------------------------------

@pytest.mark.allow_network
@pytest.mark.parametrize("model_choice,expected_dim", MODELS_TO_CACHE, ids=MODEL_IDS)
def test_create_db_model(model_choice, expected_dim, workspace):
    """
    Creates embeddings with the given model. Verifies correct dimensionality.
    This test downloads the model if not already cached.
    """
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
