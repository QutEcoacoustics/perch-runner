"""
Integration tests for run_recognizers_over_db using the koala recognizer fixture.
"""
import json
from pathlib import Path

import pyarrow.csv as pcsv
import pytest

from src.embed_export_table import run_recognizers_over_db

FIXTURES_DIR = Path("tests/files")
KOALA_CONFIG = FIXTURES_DIR / "configs" / "koala.json"
HOPLITE_PERCH_8 = FIXTURES_DIR / "hoplite_perch_8"


def _load_recognizers():
    return json.loads(KOALA_CONFIG.read_text())["recognizers"]


class TestRunRecognizersOverDbKoala:
    """Smoke-tests for run_recognizers_over_db with the real koala recognizer."""

    def test_produces_csv_output(self, workspace):
        """run_recognizers_over_db writes at least one CSV file into the output tree."""
        _, output = workspace

        run_recognizers_over_db(
            db_path=HOPLITE_PERCH_8,
            output_parent=output,
            recognizers=_load_recognizers(),
            classify_filetype="csv",
        )

        csvs = list(output.rglob("*.csv"))
        assert csvs, f"No CSV files written under {output}"

    def test_csv_has_expected_columns(self, workspace):
        """Output CSV contains the standard result columns: source, channel, offset, label, score."""
        _, output = workspace

        run_recognizers_over_db(
            db_path=HOPLITE_PERCH_8,
            output_parent=output,
            recognizers=_load_recognizers(),
            classify_filetype="csv",
        )

        csvs = list(output.rglob("*.csv"))
        assert csvs, "No CSV files written"

        table = pcsv.read_csv(csvs[0])
        for col in ("source", "channel", "offset", "label", "score"):
            assert col in table.column_names, f"Expected column '{col}' missing from {csvs[0].name}"

    def test_label_column_contains_koala(self, workspace):
        """All rows in the output label column contain 'Koala'."""
        _, output = workspace

        run_recognizers_over_db(
            db_path=HOPLITE_PERCH_8,
            output_parent=output,
            recognizers=_load_recognizers(),
            classify_filetype="csv",
        )

        csvs = list(output.rglob("*.csv"))
        assert csvs, "No CSV files written"

        table = pcsv.read_csv(csvs[0])
        labels = table.column("label").to_pylist()
        assert all(lbl == "Koala" for lbl in labels), f"Unexpected labels: {set(labels)}"
