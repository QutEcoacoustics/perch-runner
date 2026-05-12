import json
import os
from pathlib import Path

__version__ = os.environ.get("APP_VERSION", "dev")

_models_path = Path(__file__).parent / "models.json"
MODELS = json.loads(_models_path.read_text()) if _models_path.exists() else {}
