"""Integration tests for the embed(config) pipeline.

These tests run real model inference and verify exported parquet outputs.
"""

import shutil

import numpy as np
import pandas as pd

from src import embed
from src import data_frames
from src.config import EmbeddingsFormat

from .embed_helpers import FIXTURES_DIR


def test_embed_config_full_pipeline_to_parquet(workspace):
    """Run embed(config) end-to-end and validate exported parquet."""
    source, output = workspace
    one = source / "one"
    one.mkdir()
    shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", one)

    config = {
        "source": str(source),
        "output": str(output),
        "db_path": str(output / "db"),
        "model_choice": "perch_v2",
        "dataset_name": "search_set",
        "embed": [EmbeddingsFormat("parquet", "serialized")],
    }
    embed.embed(config)

    parquet_path = output / "one" / "100sec.wav" / "embeddings.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 19
    assert df.shape[1] == 4
    assert "embeddings" in df.columns
    emb = data_frames.deserialize_array(df["embeddings"].iloc[0], dtype=np.float32)
    assert len(emb) == 1536
    assert df["source"].iloc[0] == "one/100sec.wav"
    assert df["offset"].is_monotonic_increasing
