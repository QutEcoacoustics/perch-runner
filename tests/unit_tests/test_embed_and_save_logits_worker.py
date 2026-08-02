import numpy as np

from perch_hoplite.agile import embed as agile_embed
from src.embed_and_save_logits_worker import (
    LogitSavingWorker,
    _validate_class_name_mapping,
    resolve_species_class_names,
    _select_top_indices_above_threshold,
)


class _ClassNames:
    def __init__(self, classes):
        self.classes = classes


class _PresetModel:
    def __init__(self, class_list):
        self.class_list = class_list


def test_select_top_indices_applies_threshold_and_top_n():
    logits = np.array([0.50, 0.95, 0.10, 0.80, 0.70], dtype=np.float32)

    got = _select_top_indices_above_threshold(logits, threshold=0.6, top_n=2)

    # Eligible indices are [1, 3, 4], top-2 by score are [1, 3].
    assert got.tolist() == [1, 3]


def test_select_top_indices_returns_all_when_fewer_than_top_n():
    logits = np.array([0.9, 0.2, 0.7], dtype=np.float32)

    got = _select_top_indices_above_threshold(logits, threshold=0.5, top_n=10)

    assert got.tolist() == [0, 2]


def test_select_top_indices_returns_empty_when_no_scores_meet_threshold():
    logits = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    got = _select_top_indices_above_threshold(logits, threshold=0.5, top_n=10)

    assert got.size == 0


def test_select_top_indices_applies_allowed_indices_before_top_n():
    logits = np.array([0.99, 0.98, 0.97, 0.40, 0.39, 0.38], dtype=np.float32)

    # Top-2 globally would be indices [0, 1].
    # With allowed indices [2, 3, 4] applied first, top-2 should be [2, 3].
    got = _select_top_indices_above_threshold(
        logits,
        threshold=0.35,
        top_n=2,
        allowed_indices=[2, 3, 4],
    )

    assert got.tolist() == [2, 3]


def test_resolve_species_class_names_for_perch_8_prefers_label_key():
    class_list = {
        "label": _ClassNames(["abycat1", "acafly"]),
        "genus": _ClassNames(["abyg"]),
    }

    got = resolve_species_class_names(
        class_list=class_list,
        model_choice="perch_8",
        ebird_code_to_name={"abycat1": "Abyssinian Catbird", "acafly": "Acadian Flycatcher"},
    )

    assert got == ["Abyssinian Catbird", "Acadian Flycatcher"]


def test_resolve_species_class_names_for_perch_8_normalizes_ebird2021_labels_first():
    class_list = {
        "label": _ClassNames(["bkfqua1", "bkhcan1"]),
    }

    got = resolve_species_class_names(
        class_list=class_list,
        model_choice="perch_8",
        ebird_code_to_name={
            "quailf1": "ortygospiza atricollis",
            "bkhcan2": "serinus alario",
        },
        ebird_label_to_species_code={
            "bkfqua1": "quailf1",
            "bkhcan1": "bkhcan2",
        },
    )

    assert got == ["ortygospiza atricollis", "serinus alario"]


def test_resolve_species_class_names_for_perch_v2_uses_labels_key():
    class_list = {
        "labels": _ClassNames(["species_x", "species_y", "species_z"]),
    }

    got = resolve_species_class_names(
        class_list=class_list,
        model_choice="perch_v2",
        ebird_code_to_name={"species_x": "ignored"},
    )

    assert got == ["species_x", "species_y", "species_z"]


def test_logit_worker_loads_class_names_for_perch_8(monkeypatch):
    monkeypatch.setattr(
        agile_embed.EmbedWorker,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    def _load_model_by_name(model_choice):
        assert model_choice == "perch_8"
        return _PresetModel(
            {
                "label": _ClassNames(["species_a", "species_b"]),
                "genus": _ClassNames(["genus_a"]),
            }
        )

    monkeypatch.setattr(
        "perch_hoplite.zoo.model_configs.load_model_by_name",
        _load_model_by_name,
    )

    worker = LogitSavingWorker(model_choice="perch_8")

    assert worker.logits_key == "label"
    assert worker.class_names == ["species_a", "species_b"]
    assert worker.max_classes_per_segment == 10


def test_logit_worker_loads_class_names_for_perch_v2(monkeypatch):
    monkeypatch.setattr(
        agile_embed.EmbedWorker,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    def _load_model_by_name(model_choice):
        assert model_choice == "perch_v2"
        return _PresetModel(
            {
                "labels": _ClassNames(["species_x", "species_y", "species_z"]),
            }
        )

    monkeypatch.setattr(
        "perch_hoplite.zoo.model_configs.load_model_by_name",
        _load_model_by_name,
    )

    worker = LogitSavingWorker(model_choice="perch_v2")

    assert worker.logits_key == "label"
    assert worker.class_names == ["species_x", "species_y", "species_z"]


def test_logit_worker_applies_custom_max_detections_and_species_filter(monkeypatch):
    monkeypatch.setattr(
        agile_embed.EmbedWorker,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    def _load_model_by_name(_model_choice):
        return _PresetModel(
            {
                "labels": _ClassNames(["Koala", "Emu", "Currawong"]),
            }
        )

    monkeypatch.setattr(
        "perch_hoplite.zoo.model_configs.load_model_by_name",
        _load_model_by_name,
    )

    worker = LogitSavingWorker(
        model_choice="perch_v2",
        perch_max_detections_per_window=4,
        perch_species_list=["koala", "currawong"],
    )

    assert worker.max_classes_per_segment == 4
    assert worker.perch_species_filter == {"koala", "currawong"}


def test_validate_class_name_mapping_accepts_matching_lengths():
    _validate_class_name_mapping(["a", "b", "c"], 3)


def test_validate_class_name_mapping_raises_for_mismatch():
    import pytest

    with pytest.raises(ValueError, match="Mismatch between logits class dimension and resolved species names"):
        _validate_class_name_mapping(["a", "b"], 3)
