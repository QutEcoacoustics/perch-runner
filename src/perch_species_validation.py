"""Helpers for validating Perch species lists against final export labels."""

from functools import lru_cache
from pathlib import Path


SPECIES_LIST_PRESETS = {
    "australian_birds_01": "src/species_lists/national_candidate_species.txt",
}


_MODEL_SPECIES_LIST_FILES = {
    "perch_8": "perch_8.txt",
    "perch_v2": "perch_v2.txt",
}


@lru_cache(maxsize=None)
def _load_allowed_species_map(model_choice: str) -> dict[str, str]:
    filename = _MODEL_SPECIES_LIST_FILES.get(model_choice)
    if filename is None:
        raise ValueError(f"No saved species list is configured for model_choice={model_choice}.")

    list_path = Path(__file__).resolve().parent / "species_lists" / filename
    if not list_path.exists():
        raise FileNotFoundError(
            f"Saved species list for model_choice={model_choice} not found at {list_path}."
        )

    allowed_species = {}
    for line in list_path.read_text(encoding="utf-8").splitlines():
        species_name = line.strip()
        if species_name:
            allowed_species[species_name.casefold()] = species_name
    return allowed_species


def validate_perch_species_list_entries(config: dict) -> None:
    """Validate configured species names against saved final-label lists."""
    species_list = config.get("perch_species_list")
    if not species_list:
        return

    allowed_species = _load_allowed_species_map(config["model_choice"])
    canonical_species = []
    invalid_species = []
    seen = set()

    for raw_name in species_list:
        normalized_name = str(raw_name).strip().casefold()
        canonical_name = allowed_species.get(normalized_name)
        if canonical_name is None:
            invalid_species.append(str(raw_name).strip())
            continue

        if canonical_name not in seen:
            seen.add(canonical_name)
            canonical_species.append(canonical_name)

    if invalid_species:
        examples = ", ".join(repr(name) for name in invalid_species[:10])
        raise ValueError(
            "perch_species_list contains species not present in the final label set for "
            f"model_choice={config['model_choice']}: {examples}"
        )

    config["perch_species_list"] = canonical_species