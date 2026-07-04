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
    return ClassifierConfigList.from_any(json.loads(KOALA_CONFIG_PATH.read_text())["recognizers"])


def test_run_recognizers_over_db_classifies_each_source_once(tmp_path, monkeypatch):
    # Two sources each with a small number of rows; classify_table should be called once per source.
    data_by_source = {
        "site1/audio.wav": [(0.0, 1), (5.0, 2)],
        "site2/audio.wav": [(0.0, 3)],
    }

    def fake_load_db_and_group_windows(_db_path):
        return object(), data_by_source

    monkeypatch.setattr(db_to_table, "_load_db_and_group_windows", fake_load_db_and_group_windows)

    def fake_build_rows(source_value, entries, _embedding_table_format, _db):
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
        classify_filetype="csv",
        output_template="results{ext}",
    )

    # classify_table called exactly once per source (2 sources)
    assert len(classify_calls) == 2
    assert sorted(classify_calls) == [1, 2]

    output_files = sorted(tmp_path.rglob("*.csv"))
    assert len(output_files) == 1
    output_df = pd.read_csv(output_files[0])
    assert len(output_df) == 2
    assert set(output_df["label"].tolist()) == {"Koala"}
