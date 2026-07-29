import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa

import embeddings_classifier
from embeddings_classifier.app import ClassifierConfigList
from src import db_to_table


KOALA_CONFIG_PATH = Path("tests/files/configs/koala.json")


def _load_koala_recognizers():
    return ClassifierConfigList.from_any(json.loads(KOALA_CONFIG_PATH.read_text(encoding="utf-8"))["recognizers"])


def test_run_recognizers_over_db_classifies_each_source_once(tmp_path, monkeypatch):
    # Two sources each with a small number of rows; classify_table should be called once per source.
    data_by_source = {
        "site1/audio.wav": [(0.0, 1), (5.0, 2)],
        "site2/audio.wav": [(0.0, 3)],
    }

    def fake_load_db_and_group_windows(_db_path):
        return object(), data_by_source

    monkeypatch.setattr(db_to_table, "_load_db_and_group_windows", fake_load_db_and_group_windows)

    def fake_build_rows(source_value, entries, _embeddings_table_format, _db):
        return pd.DataFrame({
            "source": [source_value] * len(entries),
            "channel": [0] * len(entries),
            "offset": [e[0] for e in entries],
            "f0000": [0.1] * len(entries),
        })

    monkeypatch.setattr(db_to_table, "build_rows", fake_build_rows)

    classify_calls = []

    def fake_classify_table(table, _config, output_path=None):
        classify_calls.append(table.num_rows)
        assert output_path is None
        assert _config.configs[0].classifier_name == "koala"
        source_val = table.column("source")[0].as_py()
        result_table = pa.table({
            "source": [source_val],
            "channel": [0],
            "offset": [0.0],
            "label": ["Koala"],
            "score": [0.95],
        })
        return [
            SimpleNamespace(
                success=True,
                config=_config.configs[0],
                result_table=result_table,
                message="",
                error="",
            )
        ]

    monkeypatch.setattr(embeddings_classifier, "classify_table", fake_classify_table)

    db_to_table.run_recognizers_over_db(
        db_path=tmp_path / "fake_db",
        output_parent=tmp_path,
        recognizers=_load_koala_recognizers(),
        recognizer_results_filetype="csv",
        sourcemap=None,
        output_template="results{ext}",
        extra_columns=lambda src: {"arid": src.split("/")[0]},
    )

    # classify_table called exactly once per source (2 sources)
    assert len(classify_calls) == 2
    assert sorted(classify_calls) == [1, 2]

    output_files = sorted(tmp_path.rglob("*.csv"))
    assert len(output_files) == 1
    output_df = pd.read_csv(output_files[0])
    assert len(output_df) == 2
    assert set(output_df["label"].tolist()) == {"Koala"}
    assert set(output_df["arid"].tolist()) == {"site1", "site2"}


def test_export_classify_table_routes_rows_by_template_and_source(tmp_path, monkeypatch):
    staged = tmp_path / ".classify_staging.parquet"
    pd.DataFrame(
        {
            "window_id": ["a_0", "b_0"],
            "recording_id": ["1", "2"],
            "offset_s": [0.0, 1.0],
            "species": ["BirdA", "BirdB"],
            "score": [0.9, 0.8],
        }
    ).to_parquet(staged, index=False)

    def fake_recording_lookup(_db_path):
        return {"1": "x/site1.wav", "2": "y/site2.wav"}

    monkeypatch.setattr(db_to_table, "_load_recording_id_to_source", fake_recording_lookup)

    db_to_table.export_classify_table(
        staging_path=staged,
        db_path=tmp_path / "fake_db",
        output_path=tmp_path,
        filetype="csv",
        output_template="{parents}/{basename}/classify{ext}",
        sourcemap=None,
    )

    first = tmp_path / "x" / "site1.wav" / "classify.csv"
    second = tmp_path / "y" / "site2.wav" / "classify.csv"
    assert first.exists()
    assert second.exists()

    first_df = pd.read_csv(first)
    assert list(first_df.columns) == ["source", "channel", "offset", "label", "score"]
    assert first_df.iloc[0]["source"] == "x/site1.wav"
    assert first_df.iloc[0]["label"] == "BirdA"


def test_export_classify_table_applies_sourcemap(tmp_path, monkeypatch):
    staged = tmp_path / ".classify_staging.parquet"
    pd.DataFrame(
        {
            "window_id": ["a_0"],
            "recording_id": ["3"],
            "offset_s": [2.5],
            "species": ["BirdC"],
            "score": [0.7],
        }
    ).to_parquet(staged, index=False)

    def fake_recording_lookup(_db_path):
        return {"3": "nested/file.wav"}

    monkeypatch.setattr(db_to_table, "_load_recording_id_to_source", fake_recording_lookup)

    db_to_table.export_classify_table(
        staging_path=staged,
        db_path=tmp_path / "fake_db",
        output_path=tmp_path,
        filetype="csv",
        output_template="classify{ext}",
        sourcemap=lambda src: f"mapped::{src}",
        extra_columns=lambda src: {"arid": src.split("/")[0]},
    )

    out = tmp_path / "classify.csv"
    assert out.exists()
    out_df = pd.read_csv(out)
    assert out_df.iloc[0]["source"] == "mapped::nested/file.wav"
    assert out_df.iloc[0]["arid"] == "nested"
