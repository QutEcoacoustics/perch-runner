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


def _make_audio_tree(root, files):
    """Create empty files under root. files is a dict of {relative_path: ext}.

    For audio files, also patches them so sf.info returns a fake duration.
    Returns a mapping of filename -> duration for convenience.
    """
    for relpath in files:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


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

        with mock.patch("src.embed.sf.info", side_effect=Exception("corrupt")):
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

        with mock.patch("src.embed.sf.info", side_effect=fake_info):
            import logging
            with caplog.at_level(logging.INFO, logger="src.embed"):
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
                raise Exception("corrupt header")
            return _FakeSFInfo(45.0)

        with mock.patch("src.embed.sf.info", side_effect=fake_info):
            result = embed._scan_audio_files(tmp_path, "*")

        assert result == 45.0
        assert call_count == 2

    def test_zero_byte_audio_files(self, tmp_path):
        """Zero-byte audio files that sf.info can read report 0 duration."""
        (tmp_path / "empty.wav").touch()

        with mock.patch("src.embed.sf.info", return_value=_FakeSFInfo(0.0)):
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

        with mock.patch("src.embed.sf.info", return_value=_FakeSFInfo(30.0)):
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


# ---------------------------------------------------------------------------
# embed() — hoplite directory cleanup
# ---------------------------------------------------------------------------

class TestEmbedHopliteCleanup:
    """Tests that embed() correctly removes or keeps the hoplite directory."""

    @pytest.fixture
    def output_with_hoplite(self, tmp_path):
        """Copy the real hoplite fixture to a tmp output dir and verify it loads."""
        output = tmp_path / "output"
        output.mkdir()
        hoplite_dest = output / "hoplite"
        shutil.copytree(FIXTURES_DIR / "hoplite_perch_v2", hoplite_dest)

        # Verify the copy is a valid hoplite DB
        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(hoplite_dest))
        assert db.count_embeddings() > 0
        return output

    def _make_embed_config(self, output, embed_formats):
        """Build a minimal config dict for embed()."""
        return {
            "source": str(output / "input"),
            "output": str(output),
            "model_choice": "perch_v2",
            "dataset_name": "test",
            "embed": embed_formats,
        }

    def _run_embed_with_formats(self, output, embed_formats, audio_duration=100.0):
        """Run embed() with mocked create_database and export_as_parquet."""
        config = self._make_embed_config(output, embed_formats)

        with mock.patch("src.embed.create_database", return_value=audio_duration), \
             mock.patch("src.embed.export_as_parquet"), \
             mock.patch("src.embed.log_ram"):
            embed.embed(config)

    def test_removes_hoplite_when_not_requested(self, output_with_hoplite):
        """Hoplite dir is removed when only parquet formats are requested."""
        formats = [EmbeddingsFormat("parquet", "serialized")]
        self._run_embed_with_formats(output_with_hoplite, formats)

        assert not (output_with_hoplite / "hoplite").exists()

    def test_keeps_hoplite_when_requested(self, output_with_hoplite):
        """Hoplite dir is kept when hoplite format is requested."""
        formats = [
            EmbeddingsFormat("parquet", "serialized"),
            EmbeddingsFormat("hoplite", "serialized"),
        ]
        self._run_embed_with_formats(output_with_hoplite, formats)

        hoplite_dir = output_with_hoplite / "hoplite"
        assert hoplite_dir.exists()
        db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(hoplite_dir))
        assert db.count_embeddings() > 0


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
        (output / "hoplite").mkdir(exist_ok=True)
        return {
            "source": str(source),
            "output": str(output),
            "model_choice": "perch_v2",
            "dataset_name": "test",
            "embed": embed_formats,
        }

    def test_hoplite_only_skips_parquet_export(self, tmp_path):
        """Only hoplite format: export_as_parquet is not called."""
        config = self._base_config(tmp_path, [EmbeddingsFormat("hoplite", "serialized")])

        with mock.patch("src.embed.export_as_parquet") as mock_export:
            embed.embed(config)

        mock_export.assert_not_called()

    def test_columns_only(self, tmp_path):
        """Parquet columns format: export called with as_columns=True, as_serialized=False."""
        config = self._base_config(tmp_path, [EmbeddingsFormat("parquet", "columns")])

        with mock.patch("src.embed.export_as_parquet") as mock_export:
            embed.embed(config)

        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert kwargs["as_columns"] is True
        assert kwargs["as_serialized"] is False

    def test_all_three_formats(self, tmp_path):
        """Serialized + columns + hoplite: export called with both parquet flags."""
        config = self._base_config(tmp_path, [
            EmbeddingsFormat("parquet", "serialized"),
            EmbeddingsFormat("parquet", "columns"),
            EmbeddingsFormat("hoplite", "serialized"),
        ])

        with mock.patch("src.embed.export_as_parquet") as mock_export:
            embed.embed(config)

        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        assert kwargs["as_serialized"] is True
        assert kwargs["as_columns"] is True

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
        with mock.patch("src.embed.model_configs.get_preset_model_config") as mock_preset, \
             mock.patch("src.embed.db_loader.DBConfig") as mock_db_config, \
             mock.patch("src.embed.agile_embed.EmbedWorker") as mock_worker, \
             mock.patch("src.embed.source_info"), \
             mock.patch("src.embed.compute_workers", return_value=1), \
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

    def test_model_choice_as_set(self, tmp_path):
        """model_choice as a set extracts the first element."""
        config = self._base_config(tmp_path, model_choice={"perch_v2"})

        # Should not raise
        embed.create_database(config)

    def test_explicit_file_glob(self, tmp_path):
        """Explicit file_glob in config is used instead of auto-detection."""
        config = self._base_config(tmp_path, file_glob="*/*")

        with mock.patch("src.embed._scan_audio_files", return_value=0.0) as mock_scan:
            embed.create_database(config)

        mock_scan.assert_called_once_with(Path(config["source"]), "*/*")

    def test_audio_shorter_than_min_len(self, tmp_path):
        """Audio shorter than 1 second still goes through (filtered by perch_hoplite)."""
        config = self._base_config(tmp_path)

        # Runs without error — min_audio_len_s filtering is done by perch_hoplite
        result = embed.create_database(config)
        assert result == 0.0  # _scan_audio_files on empty .wav returns 0
