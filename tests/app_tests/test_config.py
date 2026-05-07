import argparse
from pathlib import Path

import pytest

from src.config import (
    EmbeddingsFormat,
    normalize_bool_string,
    parse_list_values,
    validate_embed_config,
    validate_value,
    load_config,
    valid_values,
)


# ---------------------------------------------------------------------------
# normalize_bool_string
# ---------------------------------------------------------------------------

class TestNormalizeBoolString:

    @pytest.mark.parametrize("value,expected", [
        (None, False),
        (False, False),
        ("none", False),
        ("None", False),
        ("false", False),
        ("False", False),
        ("null", False),
        ("", False),
        (True, True),
        ("true", True),
        ("True", True),
        ("parquet", "parquet"),
        ("csv", "csv"),
        ("parquet-columns", "parquet-columns"),
    ])
    def test_normalize(self, value, expected):
        assert normalize_bool_string(value) is expected or normalize_bool_string(value) == expected


# ---------------------------------------------------------------------------
# EmbeddingsFormat
# ---------------------------------------------------------------------------

class TestEmbeddingsFormat:

    def test_valid(self):
        ef = EmbeddingsFormat("parquet", "serialized")
        assert ef.filetype == "parquet"
        assert ef.table_format == "serialized"

    def test_valid_columns(self):
        ef = EmbeddingsFormat("csv", "columns")
        assert ef.filetype == "csv"
        assert ef.table_format == "columns"

    def test_invalid_filetype(self):
        with pytest.raises(ValueError, match="Invalid filetype"):
            EmbeddingsFormat("xlsx", "serialized")

    def test_invalid_table_format(self):
        with pytest.raises(ValueError, match="Invalid table format"):
            EmbeddingsFormat("parquet", "wide")

    def test_all_valid_filetypes(self):
        for ft in EmbeddingsFormat.valid_filetypes:
            ef = EmbeddingsFormat(ft, "serialized")
            assert ef.filetype == ft

    def test_all_valid_table_formats(self):
        for tf in EmbeddingsFormat.valid_table_formats:
            ef = EmbeddingsFormat("parquet", tf)
            assert ef.table_format == tf


# ---------------------------------------------------------------------------
# parse_list_values
# ---------------------------------------------------------------------------

class TestParseListValues:

    def test_single_string(self):
        assert parse_list_values("parquet") == {"parquet"}

    def test_comma_separated(self):
        assert parse_list_values("parquet,csv") == {"parquet", "csv"}

    def test_comma_with_spaces(self):
        assert parse_list_values("parquet , csv") == {"parquet", "csv"}

    def test_uppercased(self):
        assert parse_list_values("Parquet,CSV") == {"parquet", "csv"}

    def test_list_input(self):
        assert parse_list_values(["parquet", "csv"]) == {"parquet", "csv"}

    def test_tuple_input(self):
        assert parse_list_values(("parquet",)) == {"parquet"}

    def test_set_input(self):
        assert parse_list_values({"parquet", "csv"}) == {"parquet", "csv"}

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid type"):
            parse_list_values(123)

    def test_deduplicates(self):
        assert parse_list_values("parquet,parquet") == {"parquet"}


# ---------------------------------------------------------------------------
# validate_embed_config
# ---------------------------------------------------------------------------

class TestValidateEmbedConfig:

    # --- simple cases ---

    def test_single_filetype_default_format(self):
        """--embed parquet"""
        result = validate_embed_config("parquet", {"serialized"})
        assert len(result) == 1
        assert result[0].filetype == "parquet"
        assert result[0].table_format == "serialized"

    def test_two_filetypes_default_format(self):
        """--embed parquet,csv"""
        result = validate_embed_config("parquet,csv", {"serialized"})
        filetypes = {r.filetype for r in result}
        assert filetypes == {"parquet", "csv"}
        assert all(r.table_format == "serialized" for r in result)

    def test_explicit_hyphenated_format(self):
        """--embed parquet-columns"""
        result = validate_embed_config("parquet-columns", {"serialized"})
        assert len(result) == 1
        assert result[0].filetype == "parquet"
        assert result[0].table_format == "columns"

    def test_single_filetype_with_table_format_override(self):
        """--embed parquet --embedding_table_format columns"""
        result = validate_embed_config("parquet", {"columns"})
        assert len(result) == 1
        assert result[0].filetype == "parquet"
        assert result[0].table_format == "columns"

    # --- cross-product expansion ---

    def test_cross_product_two_formats(self):
        """--embed parquet,csv --embedding_table_format columns,serialized
        bare filetypes expand across all fallback table formats"""
        result = validate_embed_config("parquet,csv", {"columns", "serialized"})
        pairs = {(r.filetype, r.table_format) for r in result}
        assert pairs == {
            ("parquet", "columns"),
            ("parquet", "serialized"),
            ("csv", "columns"),
            ("csv", "serialized"),
        }

    def test_explicit_not_expanded(self):
        """--embed parquet-columns,csv --embedding_table_format serialized,columns
        parquet-columns is explicit (1 result), csv is bare (expanded to 2)"""
        result = validate_embed_config("parquet-columns,csv", {"serialized", "columns"})
        pairs = {(r.filetype, r.table_format) for r in result}
        assert ("parquet", "columns") in pairs
        assert ("csv", "serialized") in pairs
        assert ("csv", "columns") in pairs
        assert len(pairs) == 3

    def test_single_bare_two_fallbacks(self):
        """--embed csv --embedding_table_format serialized,columns"""
        result = validate_embed_config("csv", {"serialized", "columns"})
        pairs = {(r.filetype, r.table_format) for r in result}
        assert pairs == {("csv", "serialized"), ("csv", "columns")}

    # --- error cases ---

    def test_invalid_filetype(self):
        with pytest.raises(ValueError):
            validate_embed_config("xlsx", {"serialized"})

    def test_invalid_table_format_hyphenated(self):
        with pytest.raises(ValueError):
            validate_embed_config("parquet-wide", {"serialized"})

    def test_too_many_hyphens(self):
        with pytest.raises(ValueError, match="Invalid embed config value"):
            validate_embed_config("parquet-columns-extra", {"serialized"})


# ---------------------------------------------------------------------------
# validate_value
# ---------------------------------------------------------------------------

class TestValidateValue:

    def test_valid_model_choice(self):
        config = {"model_choice": "perch_v2"}
        result = validate_value(config, "model_choice")
        assert result == {"perch_v2"}

    def test_invalid_model_choice(self):
        config = {"model_choice": "gpt4"}
        with pytest.raises(ValueError, match="Invalid model_choice"):
            validate_value(config, "model_choice")

    def test_classify_comma_separated(self):
        config = {"classify": "parquet,csv"}
        result = validate_value(config, "classify")
        assert result == {"parquet", "csv"}

    def test_classify_invalid(self):
        config = {"classify": "excel"}
        with pytest.raises(ValueError):
            validate_value(config, "classify")

    def test_embedding_table_format_both(self):
        config = {"embedding_table_format": "serialized,columns"}
        result = validate_value(config, "embedding_table_format")
        assert result == {"serialized", "columns"}


# ---------------------------------------------------------------------------
# load_config (integration)
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_defaults_only(self, tmp_path):
        """No config file, no args — just defaults."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        args = argparse.Namespace(
            embed="parquet",
            source=str(source),
            output=str(output),
            model=None,
            classify=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=None, args=args)
        assert config["source"] == source
        assert config["output"] == output
        assert len(config["embed"]) == 1
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "serialized"

    def test_yml_config_file(self, tmp_path):
        """Load from a YAML config file."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: csv\n"
        )
        config = load_config(config_path=str(config_file))
        assert len(config["embed"]) == 1
        assert config["embed"][0].filetype == "csv"

    def test_json_config_file(self, tmp_path):
        """Load from a JSON config file."""
        import json
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "source": str(source),
            "output": str(output),
            "embed": "parquet",
        }))
        config = load_config(config_path=str(config_file))
        assert config["embed"][0].filetype == "parquet"

    def test_args_override_config(self, tmp_path):
        """CLI args take precedence over config file values."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: csv\n"
        )
        args = argparse.Namespace(
            embed="parquet-columns",
            source=None,
            output=None,
            model=None,
            classify=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=str(config_file), args=args)
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "columns"

    def test_invalid_config_key(self, tmp_path):
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: parquet\nbogus_key: true\n"
        )
        with pytest.raises(ValueError, match="Invalid config key"):
            load_config(config_path=str(config_file))

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/config.yml")

    def test_missing_source_path(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: /nonexistent/path\noutput: {output}\nembed: parquet\n"
        )
        with pytest.raises(FileNotFoundError, match="Source path"):
            load_config(config_path=str(config_file))

    def test_missing_output_path(self, tmp_path):
        source = tmp_path / "input"
        source.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: /nonexistent/path\nembed: parquet\n"
        )
        with pytest.raises(FileNotFoundError, match="Output path"):
            load_config(config_path=str(config_file))

    def test_embed_cross_product_via_load_config(self, tmp_path):
        """Full integration: embed + embedding_table_format cross-product."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\n"
            f"embed: parquet,csv\nembedding_table_format: serialized,columns\n"
        )
        config = load_config(config_path=str(config_file))
        pairs = {(r.filetype, r.table_format) for r in config["embed"]}
        assert pairs == {
            ("parquet", "columns"),
            ("parquet", "serialized"),
            ("csv", "columns"),
            ("csv", "serialized"),
        }

    def test_unsupported_config_format(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("key = 'value'")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(config_path=str(config_file))

    @pytest.mark.parametrize("glob_input,expected", [
        (None, None),
        ("None", None),
        ("none", None),
        ("false", None),
        ("False", None),
        ("*/*", "*/*"),
        ("*/*/*", "*/*/*"),
    ])
    def test_file_glob_normalization(self, tmp_path, glob_input, expected):
        """file_glob falsy strings normalize to None, real patterns pass through."""
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
            file_glob=glob_input,
        )
        config = load_config(config_path=None, args=args)
        assert config["file_glob"] is expected or config["file_glob"] == expected


class TestConfigEdgeCases:
    """Tests for malformed, empty, or wrong-type config values."""

    def test_empty_yaml_file(self, tmp_path):
        """An empty YAML file (safe_load returns None) should use defaults."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text("")

        args = argparse.Namespace(
            embed="parquet",
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format=None,
            file_glob=None,
        )
        config = load_config(config_path=str(config_file), args=args)
        assert config["model_choice"] == {"perch_v2"}

    def test_yaml_only_comments(self, tmp_path):
        """A YAML file with only comments should use defaults."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text("# this is a comment\n# another comment\n")

        args = argparse.Namespace(
            embed="parquet",
            classify=None,
            source=str(source),
            output=str(output),
            model_choice=None,
            embedding_table_format=None,
            file_glob=None,
        )
        config = load_config(config_path=str(config_file), args=args)
        assert config["model_choice"] == {"perch_v2"}

    def test_malformed_yaml(self, tmp_path):
        """Malformed YAML should raise an error."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("embed: [unclosed bracket\n  bad indent: yes\n")

        with pytest.raises(Exception):
            load_config(config_path=str(config_file))

    def test_embed_wrong_type_integer(self, tmp_path):
        """embed: 123 (integer) should raise an error."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: 123\n"
        )
        with pytest.raises((ValueError, TypeError)):
            load_config(config_path=str(config_file))

    def test_embed_wrong_type_list(self, tmp_path):
        """embed: [parquet, csv] as a YAML list should still work (or fail gracefully)."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed:\n  - parquet\n  - csv\n"
        )
        config = load_config(config_path=str(config_file))
        # Should handle list input gracefully
        filetypes = {ef.filetype for ef in config["embed"]}
        assert "parquet" in filetypes
        assert "csv" in filetypes

    def test_source_is_not_string(self, tmp_path):
        """source: 123 (not a string) should raise an error."""
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: 123\noutput: {output}\nembed: parquet\n"
        )
        with pytest.raises((TypeError, FileNotFoundError)):
            load_config(config_path=str(config_file))

    def test_model_choice_boolean(self, tmp_path):
        """model_choice: true (YAML parses as boolean) should raise."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: parquet\nmodel_choice: true\n"
        )
        with pytest.raises((ValueError, TypeError, AttributeError)):
            load_config(config_path=str(config_file))

    def test_multiple_unknown_keys(self, tmp_path):
        """Multiple unknown keys all get caught."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nfoo: bar\nbaz: 42\n"
        )
        with pytest.raises(ValueError, match="Invalid config key"):
            load_config(config_path=str(config_file))
