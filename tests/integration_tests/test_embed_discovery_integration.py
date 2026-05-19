"""
Integration tests for file discovery (mocked model — fast, no TensorFlow).

Validates how create_database() discovers files and names recordings/deployments
while using a lightweight fake embedding model.
"""
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import embed
from src.config import default_config
from perch_hoplite.db import sqlite_usearch_impl

from .embed_helpers import FIXTURES_DIR, A2O_FLAC


# ---------------------------------------------------------------------------
# Fake model (avoids loading TensorFlow entirely)
# ---------------------------------------------------------------------------

class _FakeInferenceOutputs:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.logits = None
        self.separated_audio = None
        self.batched = False
        self.frontend = None


class _FakeEmbeddingModel:
    """Returns random embeddings of the right shape."""

    sample_rate = 32000
    window_size_s = 5.0
    hop_size_s = 5.0

    def __init__(self, embedding_dim=1536):
        self.embedding_dim = embedding_dim

    @classmethod
    def from_config(cls, model_config):
        return cls()

    def embed(self, audio_array: np.ndarray):
        n_windows = max(1, len(audio_array) // (self.sample_rate * 5))
        embeddings = np.random.default_rng(42).standard_normal(
            (n_windows, 1, self.embedding_dim)
        ).astype(np.float32)
        return _FakeInferenceOutputs(embeddings)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model():
    """Patches model loading to use _FakeEmbeddingModel (no TF, no CNN)."""
    with patch(
        "perch_hoplite.zoo.model_configs.get_model_class",
        return_value=_FakeEmbeddingModel,
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFileDiscoveryIntegration:
    def test_create_db_flat_source(self, workspace):
        """Audio files at the source root are discovered and embedded."""
        source, output = workspace
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", source)

        config = {**default_config, "source": str(source), "output": str(output), "dataset_name": "search_set"}
        embed.create_database(config)

        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "db"))
        assert db.count_embeddings() > 0
        assert len(db.get_all_recordings()) == 1

    def test_auto_glob_top_level_misses_nested(self, workspace):
        """
        When audio is at the top level, auto-detection picks '*'.
        Files in subdirectories are NOT found.
        """
        source, output = workspace
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", source)
        subdir = source / "deploy1"
        subdir.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "segment.flac", subdir)

        config = {**default_config, "source": str(source), "output": str(output), "dataset_name": "search_set"}
        embed.create_database(config)

        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "db"))
        assert db.count_embeddings() > 0
        filenames = [r.filename for r in db.get_all_recordings()]
        assert any("100sec.wav" in f for f in filenames)
        assert not any("segment.flac" in f for f in filenames)

    def test_auto_glob_second_level(self, workspace):
        """When audio is one level deep, auto-detection picks '*/*'."""
        source, output = workspace
        subdir = source / "deploy1"
        subdir.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", subdir)

        config = {**default_config, "source": str(source), "output": str(output), "dataset_name": "search_set"}
        embed.create_database(config)

        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "db"))
        assert db.count_embeddings() > 0
        recordings = db.get_all_recordings()
        assert len(recordings) == 1
        assert "100sec.wav" in recordings[0].filename

    def test_a2o_flac_nested(self, workspace):
        """
        A2O-style FLAC in a subdirectory: recording filename includes site prefix,
        deployment named after subdirectory.
        """
        source, output = workspace
        site_dir = source / "Minjerribah-Dry-B"
        site_dir.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / A2O_FLAC, site_dir)

        config = {**default_config, "source": str(source), "output": str(output), "dataset_name": "search_set"}
        embed.create_database(config)

        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "db"))
        assert db.count_embeddings() > 0
        recordings = db.get_all_recordings()
        assert len(recordings) == 1
        assert recordings[0].filename == f"Minjerribah-Dry-B/{A2O_FLAC}"
        deployments = [d.name for d in db.get_all_deployments()]
        assert "Minjerribah-Dry-B" in deployments

    def test_a2o_flac_flat(self, workspace):
        """A2O-style FLAC at the source root: deployment defaults to dataset_name."""
        source, output = workspace
        shutil.copy(FIXTURES_DIR / "audio" / A2O_FLAC, source)

        config = {**default_config, "source": str(source), "output": str(output), "dataset_name": "search_set"}
        embed.create_database(config)

        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(output / "db"))
        assert db.count_embeddings() > 0
        recordings = db.get_all_recordings()
        assert len(recordings) == 1
        assert recordings[0].filename == A2O_FLAC
        deployments = [d.name for d in db.get_all_deployments()]
        assert "search_set" in deployments
