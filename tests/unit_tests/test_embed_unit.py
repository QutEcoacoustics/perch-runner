"""Unit tests for embed.py using the current single-format config model."""

from pathlib import Path
from unittest import mock

import pytest

from src import embed
from src.sourcemap import SourcemapConfig


def _base_config(tmp_path, **overrides):
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    cfg = {
        "source": source,
        "output": output,
        "db_path": output / "db",
        "model_choice": "perch_v2",
        "dataset_name": "search_set",
        "embed": True,
        "embeddings_table_format": "serialized",
        "embeddings_table_filetype": "parquet",
        "embeddings_output_path_template": "{parents}/{filestem}/{analysis}{ext}",
        "save_db": False,
        "recognizers": [],
        "recognizer_results_filetype": "csv",
        "recognizer_output_path_template": "{analysis}{ext}",
        "classify": False,
        "classify_filetype": "csv",
        "classify_output_path_template": "{analysis}{ext}",
        "sourcemap_config": None,
    }
    cfg.update(overrides)
    return cfg


class TestEmbedPipeline:
    def test_export_called_with_single_format_fields(self, tmp_path):
        config = _base_config(tmp_path)

        with mock.patch("src.embed.create_database", return_value=100.0), mock.patch(
            "src.embed.export_embeddings_table"
        ) as mock_export, mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_export.call_args
        assert kwargs["table_format"] == "serialized"
        assert kwargs["filetype"] == "parquet"
        assert kwargs["output_template"] == "{parents}/{filestem}/{analysis}{ext}"

    def test_recognizer_export_called_when_present(self, tmp_path):
        config = _base_config(tmp_path, recognizers=[object()])

        with mock.patch("src.embed.create_database", return_value=100.0), mock.patch(
            "src.embed.run_recognizers_over_db"
        ) as mock_run, mock.patch("src.embed.export_embeddings_table"), mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_run.call_args
        assert kwargs["recognizer_results_filetype"] == "csv"
        assert kwargs["output_template"] == "{analysis}{ext}"

    def test_new_db_deleted_when_save_db_false(self, tmp_path):
        config = _base_config(tmp_path)
        db_path = Path(config["db_path"])

        def _create_db(cfg):
            Path(cfg["db_path"]).mkdir(parents=True, exist_ok=True)
            return 100.0

        with mock.patch("src.embed.create_database", side_effect=_create_db), mock.patch(
            "src.embed.export_embeddings_table"
        ), mock.patch("src.embed.log_ram"):
            embed.embed(config)

        assert not db_path.exists()

    def test_existing_db_preserved_when_save_db_false(self, tmp_path):
        config = _base_config(tmp_path)
        db_path = Path(config["db_path"])
        db_path.mkdir(parents=True, exist_ok=True)
        marker = db_path / "marker.txt"
        marker.write_text("keep")

        with mock.patch("src.embed.create_database", return_value=100.0), mock.patch(
            "src.embed.export_embeddings_table"
        ), mock.patch("src.embed.log_ram"):
            embed.embed(config)

        assert db_path.exists()
        assert marker.exists()

    def test_sourcemap_passed_to_export_when_configured(self, tmp_path):
        config = _base_config(
            tmp_path,
            sourcemap_config=SourcemapConfig.from_inputs(
                sourcemap_name="ecosounds_original",
                file_metadata={"audio_recording_id": 909057},
            ),
        )

        with mock.patch("src.embed.create_database", return_value=100.0), mock.patch(
            "src.embed.export_embeddings_table"
        ) as mock_export, mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_export.call_args
        sourcemap_fn = kwargs["sourcemap"]
        assert callable(sourcemap_fn)
        mapped = sourcemap_fn("site_0277/20210428T100000Z_Five-Rivers-Dry-A_909057.flac")
        assert mapped == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_sourcemap_passed_to_recognizers_when_configured(self, tmp_path):
        config = _base_config(
            tmp_path,
            recognizers=[object()],
            sourcemap_config=SourcemapConfig.from_inputs(
                sourcemap_name="ecosounds_original",
                file_metadata={"audio_recording_id": 909057},
            ),
        )

        with mock.patch("src.embed.create_database", return_value=100.0), mock.patch(
            "src.embed.run_recognizers_over_db"
        ) as mock_run, mock.patch("src.embed.export_embeddings_table"), mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_run.call_args
        sourcemap_fn = kwargs["sourcemap"]
        assert callable(sourcemap_fn)

    def test_classify_export_called_when_classify_enabled(self, tmp_path):
        config = _base_config(tmp_path, classify=True)

        def _create_db(cfg):
            cfg["_classify_staging_path"] = Path(cfg["output"]) / ".classify_staging.parquet"
            return 100.0

        with mock.patch("src.embed.create_database", side_effect=_create_db), mock.patch(
            "src.embed.export_classify_table"
        ) as mock_classify_export, mock.patch("src.embed.export_embeddings_table"), mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_classify_export.call_args
        assert kwargs["filetype"] == "csv"
        assert kwargs["output_template"] == "{analysis}{ext}"

    def test_no_pattern_sourcemap_with_multiple_files_raises(self, tmp_path):
        config = _base_config(
            tmp_path,
            sourcemap_config=SourcemapConfig.from_inputs(
                sourcemap_template="https://api.ecosounds.org/audio_recordings/1234/original",
            ),
        )
        (config["source"] / "a.wav").write_bytes(b"x")
        (config["source"] / "b.wav").write_bytes(b"x")

        with pytest.raises(ValueError, match="sourcemap without a pattern"):
            embed.embed(config)

    def test_no_pattern_sourcemap_with_single_file_is_allowed(self, tmp_path):
        config = _base_config(
            tmp_path,
            sourcemap_config=SourcemapConfig.from_inputs(
                sourcemap_template="https://api.ecosounds.org/audio_recordings/1234/original",
            ),
        )
        single_file = config["source"] / "single.wav"
        single_file.write_bytes(b"x")
        config["source"] = single_file

        with mock.patch("src.embed.create_database", return_value=100.0), mock.patch(
            "src.embed.export_embeddings_table"
        ) as mock_export, mock.patch("src.embed.log_ram"):
            embed.embed(config)

        assert mock_export.called
