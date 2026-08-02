"""Translate external species list labels into Perch model label spaces.

Supports BirdNET-style labels like "common_name_Scientific name" or
"Scientific name_Common Name" and maps them into the labels expected by
perch_8 (eBird codes) or perch_v2 (scientific names).
"""

import logging
from typing import Iterable

from perch_hoplite.taxonomy import namespace_db

log = logging.getLogger(__name__)



SPECIES_LIST_PRESETS = {
    "australian_birds_01": "src/species_lists/national_candidate_species.txt"
}


def _candidate_labels(raw_label: str) -> list[str]:
    """Return plausible label tokens from a raw species-list entry."""
    cleaned = raw_label.strip()
    if not cleaned:
        return []

    candidates = [cleaned]
    if "_" in cleaned:
        left, right = cleaned.split("_", 1)
        left = left.strip()
        right = right.strip()
        if left:
            candidates.append(left)
        if right:
            candidates.append(right)

    # Preserve order while deduplicating.
    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _translate_for_perch_8(species_list: Iterable[str]) -> tuple[list[str], list[str]]:
    """Translate to perch_8 label space (eBird species codes)."""
    db = namespace_db.load_db()
    species_namespace = db.namespaces["ebird2021_species"].classes
    clements_to_species = db.mappings["ebird2021_clements_to_species"].mapped_pairs

    translated = []
    unmatched = []

    for item in species_list:
        found = None
        candidates = _candidate_labels(item)

        # Already an eBird species code?
        for candidate in candidates:
            if candidate in species_namespace:
                found = candidate
                break

        # Otherwise treat candidate as scientific name in clements mapping.
        if found is None:
            for candidate in candidates:
                mapped = clements_to_species.get(candidate.lower())
                if mapped is not None:
                    found = mapped
                    break

        if found is None:
            unmatched.append(item)
        else:
            translated.append(found)

    # Preserve order while deduplicating.
    translated = list(dict.fromkeys(translated))
    return translated, unmatched


def _translate_for_perch_v2(species_list: Iterable[str]) -> tuple[list[str], list[str]]:
    """Translate to perch_v2 label space (scientific names)."""
    db = namespace_db.load_db()
    valid_labels = db.namespaces["inat2024_fsd50k"].classes
    valid_by_lower = {label.lower(): label for label in valid_labels}
    inat_mappings = [
        mapping.mapped_pairs
        for _, mapping in sorted(db.mappings.items())
        if mapping.target_namespace == "inat2024"
    ]

    translated = []
    unmatched = []

    for item in species_list:
        found = None
        for candidate in _candidate_labels(item):
            if candidate in valid_labels:
                found = candidate
                break

            lower = candidate.lower()
            if lower in valid_by_lower:
                found = valid_by_lower[lower]
                break

            # Fallback: map synonym/source taxonomies into inat2024 labels.
            for mapping in inat_mappings:
                mapped = mapping.get(candidate)
                if mapped is None:
                    mapped = mapping.get(lower)
                if mapped is None:
                    continue

                if mapped in valid_labels:
                    found = mapped
                    break

                mapped_lower = mapped.lower()
                if mapped_lower in valid_by_lower:
                    found = valid_by_lower[mapped_lower]
                    break

            if found is not None:
                break

        if found is None:
            unmatched.append(item)
        else:
            translated.append(found)

    translated = list(dict.fromkeys(translated))
    return translated, unmatched


def translate_species_list_for_model(species_list: list[str], model_choice: str) -> list[str]:
    """Translate species list entries to the target model's label space.

    Raises ValueError when no entries can be translated for supported models.
    """
    if not species_list:
        return species_list

    if model_choice == "perch_8":
        translated, unmatched = _translate_for_perch_8(species_list)
    elif model_choice == "perch_v2":
        translated, unmatched = _translate_for_perch_v2(species_list)
    else:
        return species_list

    if unmatched:
        log.warning(
            "Species list translation skipped %d unmatched entries for model %s. Examples: %s",
            len(unmatched),
            model_choice,
            unmatched[:10],
        )

    if not translated:
        raise ValueError(
            "No species list entries matched the target model label space "
            f"for model_choice={model_choice}."
        )

    return translated
