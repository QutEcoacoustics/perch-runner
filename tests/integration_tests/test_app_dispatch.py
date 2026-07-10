"""Integration tests for CLI argument parsing and main dispatch."""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.app import get_parser, main
from src.config import load_config


FIXTURES_DIR = Path("tests/files")
KOALA_CONFIG = FIXTURES_DIR / "configs" / "koala.json"


class TestCLIArgsToConfig:
    """Test that CLI args flow into current config keys and normalization."""

    def test_all_cli_args_produce_expected_config(self, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        parser = get_parser()
        args = parser.parse_args(
            [
                "analyze",
                "--source",
                str(source),
                "--output",
                str(output),
                "--embed",
                "--classify",
                "--model_choice",
                "perch_8",
                "--embeddings_table_format",
                "columns",
                "--embeddings_output_path_type",
                "nested",
                "--db_path",
                "db",
                "--file_glob",
                "*/*",
                "--workers",
                "4",
                "--log_level",
                "debug",
                "--hoplite_log_level",
                "warning",
                "--tf_log_level",
                "error",
                "--log_file",
                "/tmp/test.log",
            ]
        )

        config_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in ("command", "func")})
        config = load_config(args.config_file, config_args)

        assert config["source"] == source
        assert config["output"] == output
        assert config["embed"] is True
        assert config["classify"] is True
        assert config["model_choice"] == "perch_8"
        assert config["embeddings_table_format"] == "columns"
        assert config["embeddings_output_path_type"] == "nested"
        assert config["embeddings_output_path_template"] == "{parents}/{analysis}{ext}"
        assert config["db_path"] == output / "db"
        assert config["file_glob"] == "*/*"
        assert config["workers"] == 4
        assert config["log_level"] == "DEBUG"
        assert config["hoplite_log_level"] == "WARNING"
        assert config["tf_log_level"] == "ERROR"
        assert config["log_file"] == "/tmp/test.log"

    def test_recognizers_only_dispatches_embed(self, workspace):
        source, output = workspace

        with patch("src.app.embed") as mock_embed:
            with patch(
                "sys.argv",
                ["app", "analyze", "--source", str(source), "--output", str(output), "--recognizers", str(KOALA_CONFIG)],
            ):
                main()

        mock_embed.assert_called_once()
        config = mock_embed.call_args[0][0]
        assert len(config["recognizers"]) == 1
        assert config["model_choice"] == "perch_8"


class TestMainDispatch:
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
        assert config["embed"] is True
        assert config["embeddings_table_filetype"] == "parquet"


class TestExitCodes:
    @patch("src.app.embed", side_effect=MemoryError("OOM"))
    def test_memory_error_exits_137(self, _mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--embed"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 137


class TestNetworkBlocking:
    def test_network_is_blocked_by_default(self):
        with pytest.raises(ConnectionError, match="Network access blocked"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("8.8.8.8", 53))

    def test_download_blocked_in_tests(self):
        import kagglehub

        with pytest.raises(requests.exceptions.ConnectionError, match="Network access blocked in tests. Models must be pre-cached."):
            kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/999")
