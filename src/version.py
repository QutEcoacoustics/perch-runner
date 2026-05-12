import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__version__ = os.environ.get("APP_VERSION", "dev")

try:
	PERCH_HOPLITE_VERSION = version("perch-hoplite")
except PackageNotFoundError:
	PERCH_HOPLITE_VERSION = "unknown"

_models_path = Path(__file__).parent / "models.json"
MODELS = json.loads(_models_path.read_text()) if _models_path.exists() else {}
