"""Resolve model presets from perch-hoplite and download them.

Generates src/models.json and pre-caches the kaggle models.
Used during Docker build. Can also be run manually: python -m src.download_models
"""
import json
from pathlib import Path

import kagglehub
from perch_hoplite.zoo.model_configs import get_preset_model_config

# The only list to maintain: user-facing preset names.
MODEL_PRESETS = ["perch_8", "perch_v2"]


def resolve_models():
    """Resolve preset names to kaggle paths via perch-hoplite."""
    models = {}
    for name in MODEL_PRESETS:
        info = get_preset_model_config(name)
        models[name] = {
            "kaggle": info.model_config.tfhub_path,
            "version": info.model_config.tfhub_version,
            "embedding_dim": info.embedding_dim,
        }
    return models


def main():
    models = resolve_models()

    models_path = Path(__file__).parent / "models.json"
    models_path.write_text(json.dumps(models, indent=2) + "\n")
    print(f"Wrote {models_path}")

    for name, info in models.items():
        handle = f"{info['kaggle']}/{info['version']}"
        print(f"Downloading {name}: {handle}")
        kagglehub.model_download(handle)
        print(f"  Done: {name}")


if __name__ == "__main__":
    main()
