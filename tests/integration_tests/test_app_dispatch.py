"""Integration tests for CLI argument parsing and main dispatch."""

import argparse
import json
import shutil
import socket
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.app import main, get_parser, embed as app_embed
from src.config import default_config, load_config


FIXTURES_DIR = Path("tests/files")
KOALA_CONFIG = FIXTURES_DIR / "configs" / "koala.json"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIArgsToConfig:
    """Test that all CLI args pass through to the correct config keys."""

    def test_all_cli_args_produce_expected_config(self, tmp_path):
        """One test covering every CLI arg landing in the right config key."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        parser = get_parser()
        args = parser.parse_args([
            "analyze",
            "--source", str(source),
            "--output", str(output),
            "--embed", "parquet-columns,csv",
            "--classify", "parquet",
            "--model_choice", "perch_8",
            "--embedding_table_format", "columns",
            "--embeddings_output_path_type", "nested",
            "--db_path", "db",
            "--file_glob", "*/*",
            "--workers", "4",
            "--log_level", "debug",
            "--hoplite_log_level", "warning",
            "--tf_log_level", "error",
            "--log_file", "/tmp/test.log",
        ])
        # Filter out argparse-specific fields (command, func)
        config_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in ('command', 'func')})
        config = load_config(config_args.config_file, config_args)

        # Paths
        assert config["source"] == source
        assert config["output"] == output

        # Embed formats: parquet-columns is explicit, csv gets expanded with
        # the --embedding_table_format override
        embed_pairs = {(ef.filetype, ef.table_format) for ef in config["embed"]}
        assert ("parquet", "columns") in embed_pairs
        assert ("csv", "columns") in embed_pairs

        # Classify
        assert "parquet" in config["classify"]

        # Model
        assert config["model_choice"] == "perch_8"

        # Output path templating
        assert config["embeddings_output_path_type"] == "nested"
        assert config["embeddings_output_path_template"] == "{parents}/{basename}{ext}"
        assert config["db_path"] == output / "db"

        # File glob
        assert config["file_glob"] == "*/*"

        # Workers
        assert config["workers"] == 4

        # Log levels (uppercased by config normalization)
        assert config["log_level"] == "DEBUG"
        assert config["hoplite_log_level"] == "WARNING"
        assert config["tf_log_level"] == "ERROR"
        assert config["log_file"] == "/tmp/test.log"

    def test_load_config_and_embed_with_single_recognizer(self, workspace):
        """Recognizers-only analyze flow works from CLI args through app.embed()."""
        source, output = workspace

        with patch("src.app.embed") as mock_embed:
            with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--recognizers", str(KOALA_CONFIG)]):
                main()

        mock_embed.assert_called_once()
        config = mock_embed.call_args[0][0]
        assert len(config["recognizers"]) == 1
        assert config["model_choice"] == "perch_8"

    def test_explicit_model_choice_mismatch_with_recognizer_errors(self, workspace):
        source, output = workspace

        with patch("sys.argv", [
            "app",
            "analyze",
            "--source", str(source),
            "--output", str(output),
            "--model_choice", "perch_v2",
            "--recognizers", str(KOALA_CONFIG),
        ]):
            with pytest.raises(ValueError, match="does not match recognizer embedding model"):
                main()

    def test_load_config_and_embed_with_recognizer_path_list(self, workspace, tmp_path):
        """Comma-separated recognizer config paths normalize and run through app.embed()."""
        source, output = workspace

        recognizer_payload = json.loads(KOALA_CONFIG.read_text())["recognizers"][0]
        first_cfg = tmp_path / "koala_one.json"
        second_cfg = tmp_path / "koala_two.json"
        first_cfg.write_text(json.dumps({"recognizers": [recognizer_payload]}))
        second_payload = json.loads(json.dumps(recognizer_payload))
        second_payload["name"] = "koala_2"
        second_cfg.write_text(json.dumps({"recognizers": [second_payload]}))

        with patch("src.app.embed") as mock_embed:
            with patch("sys.argv", [
                "app",
                "analyze",
                "--source", str(source),
                "--output", str(output),
                "--recognizers", f"{first_cfg},{second_cfg}",
            ]):
                main()

        mock_embed.assert_called_once()
        config = mock_embed.call_args[0][0]
        assert len(config["recognizers"]) == 2
        assert config["model_choice"] == "perch_8"

# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

class TestMainDispatch:
    """
    Test main() dispatches to embed() and handles errors.
    This tests that the main flow of the app is correct up to calling embed() with minimal arguments
    """

    @patch("src.app.embed")
    def test_calls_embed_when_flag_set(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--embed"]):
            main()

        mock_embed.assert_called_once()
        config = mock_embed.call_args[0][0]
        assert config["embed"][0].filetype == "parquet"

    @patch("src.app.embed")
    def test_single_wav_file_source_dispatch(self, mock_embed, tmp_path):
        """CLI: If source is a .wav file, embed is called and config source is a file."""
        wav_file = tmp_path / "input.wav"
        wav_file.write_bytes(b"RIFF....WAVEfmt ")  # minimal fake wav header
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(wav_file), "--output", str(output), "--embed"]):
            main()

        mock_embed.assert_called_once()
        config = mock_embed.call_args[0][0]
        assert config["source"] == wav_file
        assert config["source"].is_file()
        # file_glob is not set at config layer for single file, handled in embed_create_db

# ---------------------------------------------------------------------------
# Exit codes / Error handling
# ---------------------------------------------------------------------------

class TestExitCodes:
    """Test that errors produce correct exit codes."""

    @patch("src.app.embed", side_effect=MemoryError("OOM"))
    def test_memory_error_exits_137(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--embed"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 137

    @patch("src.app.embed", side_effect=RuntimeError("something broke"))
    def test_generic_exception_exits_1(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--embed"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Network blocking (fixture verification)
# ---------------------------------------------------------------------------

class TestNetworkBlocking:

    """
    The app itself does not block network, but our pytest fixtures do block the network to ensure that models that
    should be caches are actually cached. 

    This test simply checks that that network blocking during testing is working. 
    """

    def test_network_is_blocked_by_default(self):
        """Verify the autouse _block_network fixture is active."""
        with pytest.raises(ConnectionError, match="Network access blocked"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("8.8.8.8", 53))

    def test_download_blocked_in_tests(self):
        """Verify that kagglehub downloads fail for uncached models due to network blocking."""
        import kagglehub
        with pytest.raises(requests.exceptions.ConnectionError, match="Network access blocked in tests. Models must be pre-cached."):
            kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/999")
