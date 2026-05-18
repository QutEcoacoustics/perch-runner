import argparse
import importlib
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.app import main, get_parser
from src.config import default_config, load_config


# ---------------------------------------------------------------------------
# This module is mainly testing that cli args are correctly parsed and passed to the config, and that main() dispatches correctly.
# and non-analyze subcommands work
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


# ---------------------------------------------------------------------------
# Exit codes
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
# Module-level behavior
# ---------------------------------------------------------------------------

class TestModuleLevel:

    def test_tf_cpp_log_level_set_at_import(self):
        """TF_CPP_MIN_LOG_LEVEL is set when app module is imported."""
        # The import at the top of this file already triggers the setdefault.
        # Verify it's set (setdefault won't overwrite if already present).
        assert "TF_CPP_MIN_LOG_LEVEL" in os.environ
        # Value should be '1' (the default) or whatever was already set
        assert os.environ["TF_CPP_MIN_LOG_LEVEL"] in ("1", "2", "3")


# ---------------------------------------------------------------------------
# Version command
# ---------------------------------------------------------------------------

class TestVersionCommand:

    def test_version_prints_and_exits(self, capsys):
        with patch.dict(os.environ, {"APP_VERSION": "dev"}):
            # src.version reads APP_VERSION at import time, and src.app imports
            # that module-level value. Reload both inside this context so the
            # command reflects the patched environment.
            import src.version
            importlib.reload(src.version)
            import src.app
            importlib.reload(src.app)
            from src.app import main
            with patch("sys.argv", ["app", "version"]):
                main()
        output = capsys.readouterr().out
        assert "perch-runner dev" in output
        assert "perch-hoplite" in output
        assert "perch_8" in output
        assert "perch_v2" in output


# ---------------------------------------------------------------------------
# Config command
# ---------------------------------------------------------------------------

class TestConfigCommand:

    def test_config_prints_default_config_and_exits(self, capsys):
        with patch("sys.argv", ["app", "config"]):
            main()

        output = capsys.readouterr().out
        printed = json.loads(output)
        assert printed == default_config


# ---------------------------------------------------------------------------
# Network blocking
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
