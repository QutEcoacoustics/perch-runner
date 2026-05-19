"""Integration tests for embedding behavior across model choices."""

import shutil

import pytest

from perch_hoplite.db import sqlite_usearch_impl

from src import embed
from src.version import MODELS

from .embed_helpers import FIXTURES_DIR


@pytest.mark.parametrize("model_choice", MODELS.keys(), ids=MODELS.keys())
def test_create_db_model(model_choice, workspace):
    """Create embeddings with each model and verify embedding dimensionality."""
    expected_dim = MODELS[model_choice]["embedding_dim"]
    source, output = workspace
    site = source / "site1"
    site.mkdir()
    shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

    config = {
        "source": str(source),
        "output": str(output),
        "db_path": str(output / "db"),
        "model_choice": model_choice,
        "dataset_name": "search_set",
    }
    embed.create_database(config)

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "db"))
    assert db.count_embeddings() > 0
    assert len(db.get_all_recordings()) == 1

    windows = db.get_all_windows()
    emb = db.get_embedding(windows[0].id)
    assert emb.shape == (expected_dim,)
