"""
Unit tests for embed.py functions.

These tests mock heavy dependencies (soundfile, perch_hoplite) to run fast
without TensorFlow or real audio processing.
"""
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src import embed
from src import embed_create_db
from src.config import EmbeddingsFormat
from perch_hoplite.db import sqlite_usearch_impl

from .embed_helpers import FIXTURES_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSFInfo:
    """Mimics soundfile.info() return value."""
    def __init__(self, duration):
        self.duration = duration


# ---------------------------------------------------------------------------
# _scan_audio_files
# ---------------------------------------------------------------------------

class TestScanAudioFiles:

    def test_no_matching_files(self, tmp_path):
        """Returns 0.0 and logs warning when no audio files match."""
        result = embed._scan_audio_files(tmp_path, "*")
        assert result == 0.0

    def test_filters_non_audio_files(self, tmp_path):
        """Non-audio files (.txt) matching glob are excluded."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.csv").touch()

        result = embed._scan_audio_files(tmp_path, "*")
        assert result == 0.0

    def test_unreadable_file_logs_warning_counts_zero(self, tmp_path, caplog):
        """Unreadable audio files log a warning and count as 0 duration."""
        (tmp_path / "bad.wav").touch()

        with mock.patch("src.embed_create_db.sf.info", side_effect=embed_create_db.sf.SoundFileRuntimeError("corrupt")):
            result = embed._scan_audio_files(tmp_path, "*")

        assert result == 0.0
        assert any("Could not read" in r.message for r in caplog.records)

    def test_correct_total_and_average(self, tmp_path, caplog):
        """Correctly calculates total and average duration."""
        (tmp_path / "a.wav").touch()
        (tmp_path / "b.wav").touch()
        (tmp_path / "c.flac").touch()

        durations = {"a.wav": 10.0, "b.wav": 20.0, "c.flac": 30.0}

        def fake_info(path):
            name = Path(path).name
            return _FakeSFInfo(durations[name])

        with mock.patch("src.embed_create_db.sf.info", side_effect=fake_info):
            import logging
            with caplog.at_level(logging.INFO, logger="src.embed_create_db"):
                result = embed._scan_audio_files(tmp_path, "*")

        assert result == 60.0
        assert any("Average duration: 20.0s" in r.message for r in caplog.records)

    def test_mixed_valid_and_corrupt(self, tmp_path):
        """Mix of readable and corrupt files: corrupt counts as 0, valid counted."""
        (tmp_path / "good.wav").touch()
        (tmp_path / "bad.flac").touch()

        call_count = 0

        def fake_info(path):
            nonlocal call_count
            call_count += 1
            name = Path(path).name
            if name == "bad.flac":
                raise embed_create_db.sf.SoundFileRuntimeError("corrupt header")
            return _FakeSFInfo(45.0)

        with mock.patch("src.embed_create_db.sf.info", side_effect=fake_info):
            result = embed._scan_audio_files(tmp_path, "*")

        assert result == 45.0
        assert call_count == 2

    def test_zero_byte_audio_files(self, tmp_path):
        """Zero-byte audio files that sf.info can read report 0 duration."""
        (tmp_path / "empty.wav").touch()

        with mock.patch("src.embed_create_db.sf.info", return_value=_FakeSFInfo(0.0)):
            result = embed._scan_audio_files(tmp_path, "*")

        assert result == 0.0

    def test_symlinked_audio_files(self, tmp_path):
        """Symlinked audio files are found and measured."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_file = real_dir / "recording.wav"
        real_file.touch()

        link_dir = tmp_path / "links"
        link_dir.mkdir()
        (link_dir / "recording.wav").symlink_to(real_file)

        with mock.patch("src.embed_create_db.sf.info", return_value=_FakeSFInfo(30.0)):
            result = embed._scan_audio_files(link_dir, "*")

        assert result == 30.0


# ---------------------------------------------------------------------------
# _detect_glob_pattern
# ---------------------------------------------------------------------------

class TestDetectGlobPattern:

    def test_ignores_directories_with_audio_like_names(self, tmp_path):
        """Directories named like audio files are not matched."""
        # Create a directory with .wav extension (weird but possible)
        (tmp_path / "fake.wav").mkdir()
        # Create real audio file deeper
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "real.wav").touch()

        result = embed._detect_glob_pattern(tmp_path)
        assert result == "*/*"

    def test_mp3_extension(self, tmp_path):
        """Detects .mp3 files."""
        (tmp_path / "song.mp3").touch()
        assert embed._detect_glob_pattern(tmp_path) == "*"

    def test_ogg_extension(self, tmp_path):
        """Detects .ogg files."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "audio.ogg").touch()
        assert embed._detect_glob_pattern(tmp_path) == "*/*"

    def test_mixed_depth_uses_shallowest(self, tmp_path):
        """When depths are mixed, auto-detection chooses the shallowest depth."""
        (tmp_path / "top.wav").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.wav").touch()

        assert embed._detect_glob_pattern(tmp_path) == "*"

    def test_warns_when_deeper_files_will_be_skipped(self, tmp_path, caplog):
        """Logs a warning that counts deeper files excluded by auto-detected glob."""
        (tmp_path / "top.wav").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "a.wav").touch()
        (tmp_path / "sub" / "b.flac").touch()

        pattern = embed._detect_glob_pattern(tmp_path)

        assert pattern == "*"
        assert any("2 deeper audio file(s) will be skipped" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# embed() — db_path wiring
# ---------------------------------------------------------------------------

class TestEmbedDbPath:
    """Tests that embed() routes DB creation/export through db_path."""

    def _make_embed_config(self, tmp_path, embed_formats, db_path=None):
        source = tmp_path / "input"
        source.mkdir(exist_ok=True)
        output = tmp_path / "output"
        output.mkdir(exist_ok=True)
        resolved_db_path = db_path if db_path is not None else output / "db"
        config = {
            "source": str(source),
            "output": str(output),
            "model_choice": "perch_v2",
            "dataset_name": "test",
            "embed": embed_formats,
            "db_path": resolved_db_path,
        }
        return config

    def test_export_uses_default_db_under_output(self, tmp_path):
        config = self._make_embed_config(tmp_path, [EmbeddingsFormat("parquet", "serialized")])

        with mock.patch("src.embed.create_database", return_value=100.0), \
             mock.patch("src.embed.export_embeddings_table") as mock_export, \
             mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_export.call_args
        assert kwargs["db_path"] == Path(config["output"]) / "db"

    def test_export_uses_configured_db_path(self, tmp_path):
        config = self._make_embed_config(
            tmp_path,
            [EmbeddingsFormat("parquet", "serialized")],
            db_path=Path(tmp_path) / "custom_db",
        )

        with mock.patch("src.embed.create_database", return_value=100.0), \
             mock.patch("src.embed.export_embeddings_table") as mock_export, \
             mock.patch("src.embed.log_ram"):
            embed.embed(config)

        _, kwargs = mock_export.call_args
        assert kwargs["db_path"] == Path(tmp_path) / "custom_db"


# ---------------------------------------------------------------------------
# embed() — format dispatch
# ---------------------------------------------------------------------------

class TestEmbedFormatDispatch:
    """Tests that embed() calls export_as_parquet with correct arguments."""

    @pytest.fixture(autouse=True)
    def patch_internals(self):
        with mock.patch("src.embed.create_database", return_value=100.0), \
             mock.patch("src.embed.log_ram"):
            yield

    def _base_config(self, tmp_path, embed_formats):
        source = tmp_path / "input"
        source.mkdir(exist_ok=True)
        output = tmp_path / "output"
        output.mkdir(exist_ok=True)
        return {
            "source": str(source),
            "output": str(output),
            "db_path": str(output / "db"),
            "model_choice": "perch_v2",
            "dataset_name": "test",
            "embed": embed_formats,
        }

    def test_columns_only(self, tmp_path):
        """Parquet columns format: export called with embeddings_formats containing one column format."""
        config = self._base_config(tmp_path, [EmbeddingsFormat("parquet", "columns")])

        with mock.patch("src.embed.export_embeddings_table") as mock_export:
            embed.embed(config)

        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert len(kwargs["embeddings_formats"]) == 1
        assert kwargs["embeddings_formats"][0].filetype == "parquet"
        assert kwargs["embeddings_formats"][0].table_format == "columns"

    def test_both_parquet_formats(self, tmp_path):
        """Serialized + columns: export called with both formats."""
        config = self._base_config(tmp_path, [
            EmbeddingsFormat("parquet", "serialized"),
            EmbeddingsFormat("parquet", "columns"),
        ])

        with mock.patch("src.embed.export_embeddings_table") as mock_export:
            embed.embed(config)

        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert len(kwargs["embeddings_formats"]) == 2
        formats = {(ef.filetype, ef.table_format) for ef in kwargs["embeddings_formats"]}
        assert formats == {("parquet", "serialized"), ("parquet", "columns")}

    def test_reraises_exceptions_after_logging(self, tmp_path, caplog):
        """Exceptions from create_database are logged and re-raised."""
        config = self._base_config(tmp_path, [EmbeddingsFormat("parquet", "serialized")])

        with mock.patch("src.embed.create_database", side_effect=RuntimeError("boom")), \
             pytest.raises(RuntimeError, match="boom"):
            embed.embed(config)


# ---------------------------------------------------------------------------
# create_database edge cases
# ---------------------------------------------------------------------------

class TestCreateDatabase:

    @pytest.fixture(autouse=True)
    def _mock_heavy_deps(self):
        """Mock perch_hoplite to avoid loading TF."""
        with mock.patch("src.embed_create_db.model_configs.get_preset_model_config") as mock_preset, \
             mock.patch("src.embed_create_db.db_loader.DBConfig") as mock_db_config, \
             mock.patch("src.embed_create_db.agile_embed.EmbedWorker") as mock_worker, \
             mock.patch("src.embed_create_db.source_info"), \
             mock.patch("src.embed_create_db.compute_workers", return_value=1), \
             mock.patch("src.embed.log_ram"):

            # Set up preset mock
            mock_preset.return_value = mock.MagicMock(
                embedding_dim=1536,
                model_key="test_model",
                model_config={},
            )

            # Set up DB mock
            mock_db = mock.MagicMock()
            mock_db.get_all_windows.return_value = []
            mock_db_config.return_value.load_db.return_value = mock_db

            # Set up worker mock
            mock_worker.return_value.process_all.return_value = None

            self.mock_db = mock_db
            self.mock_worker = mock_worker
            yield

    def _base_config(self, tmp_path, **overrides):
        source = tmp_path / "input"
        source.mkdir(exist_ok=True)
        output = tmp_path / "output"
        output.mkdir(exist_ok=True)
        (source / "test.wav").touch()
        config = {
            "source": str(source),
            "output": str(output),
            "db_path": str(output / "db"),
            "model_choice": "perch_v2",
            "dataset_name": "search_set",
        }
        config.update(overrides)
        return config

    def test_logs_warning_on_zero_embeddings(self, tmp_path, caplog):
        """Logs warning when no embeddings are produced."""
        config = self._base_config(tmp_path)

        import logging
        with caplog.at_level(logging.WARNING, logger="src.embed"):
            embed.create_database(config)

        assert any("0 embeddings" in r.message for r in caplog.records)

    def test_explicit_file_glob(self, tmp_path):
        """Explicit file_glob in config is used instead of auto-detection."""
        config = self._base_config(tmp_path, file_glob="*/*")

        with mock.patch("src.embed_create_db._scan_audio_files", return_value=0.0) as mock_scan:
            embed.create_database(config)

        mock_scan.assert_called_once_with(
            Path(config["source"]),
            "*/*",
            discovered_audio_files=None,
        )

    def test_audio_shorter_than_min_len(self, tmp_path):
        """Audio shorter than 1 second still goes through (filtered by perch_hoplite)."""
        config = self._base_config(tmp_path)

        # Runs without error — min_audio_len_s filtering is done by perch_hoplite
        result = embed.create_database(config)
        assert result == 0.0  # _scan_audio_files on empty .wav returns 0

    def test_source_file_forces_filename_glob(self, tmp_path):
        """When source is a file, base_path is parent and file_glob is filename."""
        source_dir = tmp_path / "input"
        source_dir.mkdir()
        target_file = source_dir / "justthisonefile.wav"
        target_file.touch()
        (source_dir / "other.wav").touch()

        output = tmp_path / "output"
        output.mkdir()
        config = {
            "source": str(target_file),
            "output": str(output),
            "db_path": str(output / "db"),
            "model_choice": "perch_v2",
            "dataset_name": "search_set",
            "file_glob": "*",  # should be ignored for single-file source
        }

        with mock.patch("src.embed_create_db._scan_audio_files", return_value=0.0) as mock_scan:
            embed.create_database(config)

        mock_scan.assert_called_once_with(
            source_dir,
            "justthisonefile.wav",
            discovered_audio_files=None,
        )

        kwargs = embed_create_db.source_info.AudioSourceConfig.call_args.kwargs
        assert kwargs["base_path"] == str(source_dir)
        assert kwargs["file_glob"] == "justthisonefile.wav"
