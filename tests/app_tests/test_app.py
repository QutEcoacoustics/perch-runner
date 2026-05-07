import argparse
from unittest.mock import patch, MagicMock

import pytest

from src.app import main
from src.config import load_config


class TestArgParsing:
    """Test that CLI args are parsed and merged into config correctly."""

    def test_defaults_no_embed(self, tmp_path):
        """With no --embed, embed defaults to False (no embedding)."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed=None,
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert config["embed"] == []
        assert config["classify"] == set()

    def test_bare_embed_flag(self, tmp_path):
        """--embed with no value → True → resolves to parquet."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed=True,
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert len(config["embed"]) == 1
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "serialized"

    def test_embed_parquet(self, tmp_path):
        """Explicitly passing --embed parquet enables embedding."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed="parquet",
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert len(config["embed"]) == 1
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "serialized"

    def test_embed_override(self, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed="csv",
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert config["embed"][0].filetype == "csv"

    def test_embed_with_table_format(self, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed="parquet",
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format="columns",
        )
        config = load_config(config_path=None, args=args)
        assert config["embed"][0].table_format == "columns"

    def test_model_choice_override(self, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed="parquet",
            classify=None,
            source=str(source),
            output=str(output),
            model_choice="perch_8",
            embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert config["model_choice"] == {"perch_8"}


class TestMainEntrypoint:
    """Test main() dispatches to embed() correctly."""

    @patch("src.app.embed")
    def test_main_calls_embed(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", str(source), "--output", str(output), "--embed"],
        ):
            main()

        mock_embed.assert_called_once()
        config = mock_embed.call_args[0][0]
        assert config["source"] == source
        assert config["output"] == output
        assert len(config["embed"]) >= 1
        assert config["embed"][0].filetype == "parquet"

    @patch("src.app.embed")
    def test_main_no_embed_flag_skips(self, mock_embed, tmp_path):
        """When --embed is not passed at all, embed() should not be called."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", str(source), "--output", str(output)],
        ):
            main()

        mock_embed.assert_not_called()

    @patch("src.app.embed")
    def test_main_with_embed_format(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", str(source), "--output", str(output), "--embed", "csv"],
        ):
            main()

        config = mock_embed.call_args[0][0]
        assert config["embed"][0].filetype == "csv"

    @patch("src.app.embed")
    def test_main_embed_none_skips(self, mock_embed, tmp_path):
        """--embed none disables embedding."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", str(source), "--output", str(output), "--embed", "none"],
        ):
            main()

        mock_embed.assert_not_called()

    @patch("src.app.embed")
    def test_main_embed_false_skips(self, mock_embed, tmp_path):
        """--embed false disables embedding."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", str(source), "--output", str(output), "--embed", "false"],
        ):
            main()

        mock_embed.assert_not_called()

    @patch("src.app.embed")
    def test_main_embed_none_with_classify(self, mock_embed, tmp_path):
        """When embed is 'none' but classify is set, embed() is not called."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", str(source), "--output", str(output),
             "--embed", "none", "--classify"],
        ):
            main()

        mock_embed.assert_not_called()

    def test_main_missing_source(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()

        with patch(
            "sys.argv",
            ["app", "--source", "/nonexistent/path", "--output", str(output), "--embed"],
        ):
            with pytest.raises(FileNotFoundError, match="Source path"):
                main()

    @patch("src.app.embed")
    def test_main_with_config_file(self, mock_embed, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: parquet-columns\n"
        )

        with patch(
            "sys.argv",
            ["app", "--config_file", str(config_file)],
        ):
            main()

        config = mock_embed.call_args[0][0]
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "columns"
