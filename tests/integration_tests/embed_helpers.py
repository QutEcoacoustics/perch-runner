"""
Shared constants and fixtures for embed tests.

This module is the single source of truth for:
- FIXTURE_DBS: pre-generated hoplite database paths
- Common test fixtures (workspace, mock_model)
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.version import MODELS


FIXTURES_DIR = Path("tests/files")
A2O_FLAC = "20220502T075930+1000_Minjerribah-Dry-B_1088507.flac"

# ---------------------------------------------------------------------------
# Fixture database registry: pre-generated and committed to repo.
# ---------------------------------------------------------------------------

# Map model names to their fixture DB paths (pre-generated).
# To regenerate: python -m tests.generate_fixtures
FIXTURE_DBS = {
    "perch_v2": FIXTURES_DIR / "hoplite_perch_v2",
    "perch_8": FIXTURES_DIR / "hoplite_perch_8",
}

# When this helper loads, we validate that every model has a fixture DB
_missing_fixtures = set(MODELS.keys()) - set(FIXTURE_DBS.keys())
if _missing_fixtures:
    raise ValueError(
        f"Missing fixture databases for models: {sorted(_missing_fixtures)}. "
        f"Run: python -m tests.generate_fixtures"
    )
