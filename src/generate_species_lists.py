"""Generate Perch species-list text files under src/species_lists.

Usage:
  python -m src.generate_species_lists
  python -m src.generate_species_lists --models perch_8
  python -m src.generate_species_lists --output-dir src/species_lists


This is a helper script to pre-generate species list files for Perch models.
We run this during development and check the resulting files into source control. 

"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embed_and_save_logits_worker import resolve_species_class_names_for_model_choice


MODEL_PRESETS = ["perch_8", "perch_v2"]


def generate_species_list_for_model(model_choice: str) -> list[str]:
    """Return logits-order species labels for the requested model."""
    return [str(label) for label in resolve_species_class_names_for_model_choice(model_choice)]


def write_species_list_file(output_dir: Path, model_choice: str, labels: list[str]) -> Path:
    """Write one species list file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_choice}.txt"
    output_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate species-list files for Perch models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_PRESETS,
        help="Model names to generate (default: perch_8 perch_v2)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "species_lists"),
        help="Directory where <model>.txt files are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for model_choice in args.models:
        labels = generate_species_list_for_model(model_choice)
        output_path = write_species_list_file(output_dir, model_choice, labels)
        print(f"Wrote {output_path} ({len(labels)} labels)")


if __name__ == "__main__":
    main()
