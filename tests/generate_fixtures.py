"""Generate fixture hoplite databases for each model in models.json.

Usage:
    python -m tests.generate_fixtures

This embeds a short audio file with each model and saves the resulting
hoplite database to tests/files/hoplite_<model_name>/.

Prerequisites:
  - Models must be downloaded first: python -m src.download_models
  - Runs real inference, so expect ~30s per model.

After running, update FIXTURE_DBS in tests/app_tests/embed_helpers.py
if you added a new model.
"""
import shutil
from pathlib import Path

from src import embed
from src.version import MODELS

FIXTURES_DIR = Path("tests/files")
AUDIO_FILE = "100sec.wav"


def main():
    for model_name in MODELS:
        db_dir = FIXTURES_DIR / f"hoplite_{model_name}"

        if db_dir.exists():
            print(f"Skipping {model_name}: {db_dir} already exists (delete it to regenerate)")
            continue

        print(f"Generating fixture DB for {model_name}...")

        # Use a temp workspace so create_database can discover audio files
        work_dir = Path("/tmp") / f"fixture_gen_{model_name}"
        if work_dir.exists():
            shutil.rmtree(work_dir)

        source = work_dir / "input" / "site1"
        source.mkdir(parents=True)
        output = work_dir / "output"
        output.mkdir(parents=True)

        shutil.copy(FIXTURES_DIR / "audio" / AUDIO_FILE, source)

        config = {
            "source": str(source.parent),
            "output": str(output),
            "model_choice": model_name,
            "dataset_name": "search_set",
        }
        embed.create_database(config)

        # Copy the generated DB to fixtures
        shutil.copytree(output / "hoplite", db_dir)
        shutil.rmtree(work_dir)

        print(f"  Created {db_dir}")

    print("\nDone. If you added a new model, update FIXTURE_DBS in tests/app_tests/embed_helpers.py")


if __name__ == "__main__":
    main()
