import argparse
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.app import main
from src.config import default_config, load_config


# ---------------------------------------------------------------------------
# CLI args → config integration
# ---------------------------------------------------------------------------

class TestCLIArgsToConfig:
    """Test that all CLI args pass through to the correct config keys."""

    def test_all_cli_args_produce_expected_config(self, tmp_path):
        """One test covering every CLI arg landing in the right config key."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        # Parse args the same way main() does, but test load_config directly.
        parser = argparse.ArgumentParser()
        parser.add_argument("--embed", nargs='?', const=True, default=None)
        parser.add_argument("--classify", nargs='?', const=True, default=None)
        parser.add_argument("--source", default=None)
        parser.add_argument("--output", default=None)
        parser.add_argument("--config_file", default=None)
        parser.add_argument("--model_choice", default=None)
        parser.add_argument("--embedding_table_format", default=None)
        parser.add_argument("--embeddings_output_path_template", default=None)
        parser.add_argument("--embeddings_output_path_type", default=None)
        parser.add_argument("--db_path", default=None)
        parser.add_argument("--file_glob", default=None)
        parser.add_argument("--workers", default=None)
        parser.add_argument("--log_level", default=None)
        parser.add_argument("--hoplite_log_level", default=None)
        parser.add_argument("--tf_log_level", default=None)
        parser.add_argument("--log_file", default=None)
        args = parser.parse_args([
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
        config = load_config(config_path=None, args=args)

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

    def test_bare_embed_flag_defaults_to_parquet(self, tmp_path):
        """--embed with no value resolves to parquet-serialized."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed=True, classify=None,
            source=str(source), output=str(output),
            model_choice=None, embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert len(config["embed"]) == 1
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "serialized"


    def test_no_flags_produces_error(self, tmp_path):
        """No --embed or --classify flags: load_config should raise ValueError."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed=None, classify=None,
            source=str(source), output=str(output),
            model_choice=None, embedding_table_format=None,
        )
        with pytest.raises(ValueError, match="At least one of"):
            load_config(config_path=None, args=args)


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

class TestMainDispatch:
    """Test main() dispatches to embed() and handles errors."""

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
    def test_skips_embed_when_no_flag(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        # No --embed or --classify should error during config validation
        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output)]):
            with pytest.raises(ValueError, match="At least one of"):
                main()
        mock_embed.assert_not_called()

    @patch("src.app.embed")
    def test_embed_none_disables(self, mock_embed, tmp_path):
        """--embed none disables embedding. Without --classify, config validation should error."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--embed", "none"]):
            with pytest.raises(ValueError, match="At least one of"):
                main()
        mock_embed.assert_not_called()

    @patch("src.app.embed")
    def test_embed_false_disables(self, mock_embed, tmp_path):
        """--embed false disables embedding. Without --classify, config validation should error."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output), "--embed", "false"]):
            with pytest.raises(ValueError, match="At least one of"):
                main()
        mock_embed.assert_not_called()

    @patch("src.app.embed")
    def test_config_file_used(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: parquet-columns\n"
        )

        with patch("sys.argv", ["app", "analyze", "--config_file", str(config_file)]):
            main()

        config = mock_embed.call_args[0][0]
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "columns"


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    """Test that errors produce correct exit codes."""

    def test_no_embed_or_classify_exits_1(self, tmp_path):
        """Config validation should error when neither --embed nor --classify is specified."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        # No --embed or --classify flag
        with patch("sys.argv", ["app", "analyze", "--source", str(source), "--output", str(output)]):
            with pytest.raises(ValueError, match="At least one of"):
                main()

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

    def test_missing_source_raises_before_embed(self, tmp_path):
        """Invalid paths are caught during config loading, not embed."""
        output = tmp_path / "output"
        output.mkdir()

        with patch("sys.argv", ["app", "analyze", "--source", "/nonexistent/path", "--output", str(output), "--embed"]):
            with pytest.raises(FileNotFoundError, match="Source path"):
                main()


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
        from unittest.mock import patch
        import os
        import importlib
        with patch.dict(os.environ, {"APP_VERSION": "dev"}):
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

    def test_version_does_not_call_embed(self, capsys):
        with patch("sys.argv", ["app", "version"]), patch("src.app.embed") as mock_embed:
            main()
        mock_embed.assert_not_called()

    def test_version_skips_config_loading(self):
        with patch("sys.argv", ["app", "version"]), patch("src.app.load_config") as mock_config:
            main()
        mock_config.assert_not_called()


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

    def test_config_does_not_call_embed(self):
        with patch("sys.argv", ["app", "config"]), patch("src.app.embed") as mock_embed:
            main()
        mock_embed.assert_not_called()

    def test_config_skips_config_loading(self):
        with patch("sys.argv", ["app", "config"]), patch("src.app.load_config") as mock_config:
            main()
        mock_config.assert_not_called()


# ---------------------------------------------------------------------------
# Network blocking
# ---------------------------------------------------------------------------

class TestNetworkBlocking:

    def test_network_is_blocked_by_default(self):
        """Verify the autouse _block_network fixture is active."""
        with pytest.raises(ConnectionError, match="Network access blocked"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("8.8.8.8", 53))

    def test_download_blocked_in_tests(self):
        """Verify that kagglehub downloads fail for uncached models due to network blocking."""
        import kagglehub
        with pytest.raises(ConnectionError, match="Network access blocked"):
            kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/bird-vocalization-classifier/999")
