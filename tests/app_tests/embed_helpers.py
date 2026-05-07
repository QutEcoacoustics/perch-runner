"""
Shared constants and fixtures for embed tests.

This module is the single source of truth for:
- MODELS_TO_CACHE: list of (model_choice, embedding_dim) tuples
- FIXTURE_DBS: pre-generated hoplite database paths
- Common test fixtures (workspace, mock_model)
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


FIXTURES_DIR = Path("tests/files")
A2O_FLAC = "20220502T075930+1000_Minjerribah-Dry-B_1088507.flac"

# ---------------------------------------------------------------------------
# Model registry: single source of truth for models to cache and test.
# ---------------------------------------------------------------------------

MODELS_TO_CACHE = [
    # (model_choice, expected_embedding_dim)
    ("perch_v2", 1536),
    ("perch_8", 1280),
]

MODEL_IDS = [m[0] for m in MODELS_TO_CACHE]

# Map model names to their fixture DB paths (pre-generated).
FIXTURE_DBS = {
    "perch_v2": FIXTURES_DIR / "hoplite",
    "perch_8": FIXTURES_DIR / "hoplite_perch_8",
}
