"""Unit tests for app module-level initialization and parser surface."""

import os
import argparse

# Import app module to trigger TF_CPP_MIN_LOG_LEVEL setup
from src import app  # noqa: F401
from src.config import default_config


class TestModuleLevel:

    def test_tf_cpp_log_level_set_at_import(self):
        """TF_CPP_MIN_LOG_LEVEL is set when app module is imported."""
        # The import at the top of this file already triggers the setdefault.
        # Verify it's set (setdefault won't overwrite if already present).
        assert "TF_CPP_MIN_LOG_LEVEL" in os.environ
        # Value should be '1' (the default) or whatever was already set
        assert os.environ["TF_CPP_MIN_LOG_LEVEL"] in ("1", "2", "3")


class TestAnalyzeParserSurface:

    def test_analyze_parser_keys_match_config_keys(self):
        """Parser args should match config keys, except for config_file."""
        parser = app.get_parser()
        subparsers_action = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        analyze_parser = subparsers_action.choices["analyze"]

        analyze_dests = {
            action.dest
            for action in analyze_parser._actions
            if action.dest != "help"
        }

        expected_dests = set(default_config.keys()) | {"config_file"}
        assert analyze_dests == expected_dests

    def test_parser_accepts_path_type_and_dataset_flags(self, tmp_path):
        """Analyze parser accepts the output path and dataset flags passed to config."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        parser = app.get_parser()
        args = parser.parse_args([
            "analyze",
            "--source", str(source),
            "--output", str(output),
            "--embed",
            "--classify_output_path_template", "{analysis}{ext}",
            "--classify_output_path_type", "flat",
            "--output_path_type", "nested",
            "--dataset_name", "demo_set",
        ])

        assert args.classify_output_path_template == "{analysis}{ext}"
        assert args.classify_output_path_type == "flat"
        assert args.output_path_type == "nested"
        assert args.dataset_name == "demo_set"
