"""Integration tests for recognizer and classify export flows."""

import json
import shutil
from pathlib import Path

import pyarrow.csv as pcsv
import pytest
from embeddings_classifier.app import ClassifierConfigList

from src import embed
from src.db_to_table import run_recognizers_over_db

FIXTURES_DIR = Path("tests/files")
KOALA_CONFIG = FIXTURES_DIR / "configs" / "koala.json"
HOPLITE_PERCH_8 = FIXTURES_DIR / "hoplite_perch_8"


def _load_recognizers():
    return ClassifierConfigList.from_any(json.loads(KOALA_CONFIG.read_text())["recognizers"])


def test_embed_pipeline_creates_recognizer_output_csv(workspace):
    """Run embed(config) and assert recognizer output is produced."""
    source, output = workspace

    shutil.copy(FIXTURES_DIR / "audio" / "segment.flac", source / "segment.flac")

    config = {
        "source": str(source),
        "output": str(output),
        "db_path": str(output / "db"),
        "model_choice": "perch_8",
        "dataset_name": "search_set",
        "embed": False,
        "recognizers": _load_recognizers(),
        "recognizer_results_filetype": "csv",
        "recognizer_output_path_template": "{classifier_name}/results{ext}",
    }

    embed.embed(config)

    recognizer_csv = output / "koala" / "results.csv"
    assert recognizer_csv.exists(), f"Expected recognizer output CSV at {recognizer_csv}"

    table = pcsv.read_csv(recognizer_csv)
    for col in ("source", "channel", "offset", "label", "score"):
        assert col in table.column_names, f"Expected column '{col}' missing from {recognizer_csv.name}"


@pytest.mark.parametrize("model_choice", ["perch_8", "perch_v2"])
def test_embed_pipeline_creates_classify_output_csv_with_templated_path(workspace, model_choice):
    """Run embed(config) and assert classify output is produced at templated path."""
    source, output = workspace

    nested = source / "sub"
    nested.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "audio" / "segment.flac", nested / "segment.flac")

    config = {
        "source": str(source),
        "output": str(output),
        "db_path": str(output / "db"),
        "model_choice": model_choice,
        "dataset_name": "search_set",
        "embed": False,
        "classify": True,
        "classify_filetype": "csv",
        "classify_output_path_template": "{parents}/{basename}/classify_results{ext}",
        "recognizers": [],
    }

    embed.embed(config)

    classify_csv = output / "sub" / "segment.flac" / "classify_results.csv"
    assert classify_csv.exists(), f"Expected classify output CSV at {classify_csv}"

    table = pcsv.read_csv(classify_csv)
    for col in ("source", "channel", "offset", "label", "score"):
        assert col in table.column_names, f"Expected column '{col}' missing from {classify_csv.name}"

    labels = [str(v) for v in table.column("label").to_pylist()]
    assert labels, "Expected at least one classify row"
    assert not any(label.startswith("class_") for label in labels), (
        "Expected semantic class names, got fallback class_#### labels"
    )


class TestRunRecognizersOverDbKoala:
    def test_produces_csv_output(self, workspace):
        _, output = workspace

        run_recognizers_over_db(
            db_path=HOPLITE_PERCH_8,
            output_parent=output,
            recognizers=_load_recognizers(),
            recognizer_results_filetype="csv",
            sourcemap=None,
            output_template="{classifier_name}/{parents}/{basename}/{analysis}{ext}",
        )

        csvs = list(output.rglob("*.csv"))
        assert csvs, f"No CSV files written under {output}"

    def test_csv_has_expected_columns(self, workspace):
        _, output = workspace

        run_recognizers_over_db(
            db_path=HOPLITE_PERCH_8,
            output_parent=output,
            recognizers=_load_recognizers(),
            recognizer_results_filetype="csv",
            sourcemap=None,
            output_template="{classifier_name}/{parents}/{basename}/{analysis}{ext}",
        )

        csvs = list(output.rglob("*.csv"))
        assert csvs, "No CSV files written"

        table = pcsv.read_csv(csvs[0])
        for col in ("source", "channel", "offset", "label", "score"):
            assert col in table.column_names, f"Expected column '{col}' missing from {csvs[0].name}"

    def test_label_column_contains_koala(self, workspace):
        _, output = workspace

        run_recognizers_over_db(
            db_path=HOPLITE_PERCH_8,
            output_parent=output,
            recognizers=_load_recognizers(),
            recognizer_results_filetype="csv",
            sourcemap=None,
            output_template="{classifier_name}/{parents}/{basename}/{analysis}{ext}",
        )

        csvs = list(output.rglob("*.csv"))
        assert csvs, "No CSV files written"

        table = pcsv.read_csv(csvs[0])
        labels = table.column("label").to_pylist()
        assert all(lbl == "Koala" for lbl in labels), f"Unexpected labels: {set(labels)}"
