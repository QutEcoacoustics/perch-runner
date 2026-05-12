import argparse
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from src.config import (
    EmbeddingsFormat,
    normalize_bool_string,
    parse_list_values,
    validate_embed_config,
    validate_value,
    load_config,
    find_config,
    valid_values,
)


@pytest.fixture
def make_config(tmp_path):
    """Fixture factory: creates source/output dirs and a config file.

    Returns a callable that accepts an optional dict of extra config keys
    and a format ("yml" or "json"). Returns the config file path as a string.
    """
    def _create(extra=None, format="yml"):
        source = tmp_path / "input"
        source.mkdir(exist_ok=True)
        output = tmp_path / "output"
        output.mkdir(exist_ok=True)
        data = {"source": str(source), "output": str(output)}
        if extra:
            data.update(extra)
        config_file = tmp_path / f"config.{format}"
        if format == "json":
            config_file.write_text(json.dumps(data))
        else:
            config_file.write_text(yaml.dump(data))
        return str(config_file)
    return _create


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
        assert normalize_bool_string(value) == expected


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

    def test_yml_config_file(self, make_config):
        """Load from a YAML config file."""
        path = make_config({"embed": "csv"})
        config = load_config(config_path=path)
        assert len(config["embed"]) == 1
        assert config["embed"][0].filetype == "csv"

    def test_json_config_file(self, make_config):
        """Load from a JSON config file."""
        path = make_config({"embed": "parquet"}, format="json")
        config = load_config(config_path=path)
        assert config["embed"][0].filetype == "parquet"

    def test_args_override_config(self, make_config):
        """CLI args take precedence over config file values."""
        path = make_config({"embed": "csv"})
        args = argparse.Namespace(
            embed="parquet-columns",
            source=None,
            output=None,
            model=None,
            classify=None,
            embedding_table_format=None,
        )
        config = load_config(config_path=path, args=args)
        assert config["embed"][0].filetype == "parquet"
        assert config["embed"][0].table_format == "columns"

    def test_invalid_config_key(self, make_config):
        path = make_config({"embed": "parquet", "bogus_key": True})
        with pytest.raises(ValueError, match="Invalid config key"):
            load_config(config_path=path)

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

    def test_embed_cross_product_via_load_config(self, make_config):
        """Full integration: embed + embedding_table_format cross-product."""
        path = make_config({"embed": "parquet,csv",
                           "embedding_table_format": "serialized,columns"})
        config = load_config(config_path=path)
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
        assert config["file_glob"] == expected


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

    def test_embed_wrong_type_integer(self, make_config):
        """embed: 123 (integer) should raise an error."""
        path = make_config({"embed": 123})
        with pytest.raises((ValueError, TypeError)):
            load_config(config_path=path)

    def test_embed_wrong_type_list(self, make_config):
        """embed: [parquet, csv] as a YAML list should still work (or fail gracefully)."""
        path = make_config({"embed": ["parquet", "csv"]})
        config = load_config(config_path=path)
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

    def test_model_choice_boolean(self, make_config):
        """model_choice: true (YAML parses as boolean) should raise."""
        path = make_config({"embed": "parquet", "model_choice": True})
        with pytest.raises((ValueError, TypeError, AttributeError)):
            load_config(config_path=path)

    def test_multiple_unknown_keys(self, make_config):
        """Multiple unknown keys all get caught."""
        path = make_config({"foo": "bar", "baz": 42})
        with pytest.raises(ValueError, match="Invalid config key"):
            load_config(config_path=path)


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------

class TestFindConfig:

    @pytest.fixture(autouse=True)
    def mock_config_dir(self, tmp_path):
        with mock.patch('src.config.default_config_dir', str(tmp_path)):
            yield

    def test_no_files_returns_none(self, tmp_path):
        """No config files in default dir returns None."""
        assert find_config() is None

    def test_multiple_files_raises(self, tmp_path):
        """Multiple config files raises FileExistsError."""
        (tmp_path / "config.yml").write_text("embed: parquet")
        (tmp_path / "config.yaml").write_text("embed: csv")
        with pytest.raises(FileExistsError, match="Multiple config files"):
            find_config()

    def test_finds_yaml_extension(self, tmp_path):
        """Detects a .yaml file."""
        (tmp_path / "config.yaml").write_text("embed: parquet")
        result = find_config()
        assert result is not None
        assert result.name == "config.yaml"

    def test_finds_and_parses_json(self, tmp_path):
        """Detects a .json file and load_config can parse it."""
        source = tmp_path / "input"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        (tmp_path / "config.json").write_text(json.dumps({
            "source": str(source),
            "output": str(output),
            "embed": "parquet",
        }))
        result = find_config()
        assert result is not None
        assert result.name == "config.json"
        config = load_config()
        assert config["embed"][0].filetype == "parquet"


# ---------------------------------------------------------------------------
# workers normalization
# ---------------------------------------------------------------------------

class TestWorkersNormalization:

    def test_auto_stays_string(self, make_config):
        """'auto' stays as 'auto'."""
        path = make_config({"workers": "auto"})
        config = load_config(config_path=path)
        assert config["workers"] == "auto"

    def test_numeric_string_becomes_int(self, make_config):
        """'3' becomes int 3."""
        path = make_config({"workers": "3"})
        config = load_config(config_path=path)
        assert config["workers"] == 3

    def test_non_numeric_string_falls_back(self, make_config):
        """Non-numeric string falls back to 'auto'."""
        path = make_config({"workers": "fast"})
        config = load_config(config_path=path)
        assert config["workers"] == "auto"

    def test_int_stays_int(self, make_config):
        """Integer value stays as int."""
        path = make_config({"workers": 4})
        config = load_config(config_path=path)
        assert config["workers"] == 4


# ---------------------------------------------------------------------------
# log level normalization
# ---------------------------------------------------------------------------

class TestLogConfigNormalization:

    def test_log_levels_uppercased(self, make_config):
        """Log levels are uppercased."""
        path = make_config({"log_level": "debug",
                           "hoplite_log_level": "info",
                           "tf_log_level": "warning"})
        config = load_config(config_path=path)
        assert config["log_level"] == "DEBUG"
        assert config["hoplite_log_level"] == "INFO"
        assert config["tf_log_level"] == "WARNING"

    @pytest.mark.parametrize("val", ["none", "false", "None", "False"])
    def test_log_file_falsy_becomes_none(self, make_config, val):
        """Falsy log_file values become None."""
        path = make_config({"log_file": val})
        config = load_config(config_path=path)
        assert config["log_file"] is None

    def test_log_file_real_path_preserved(self, make_config):
        """A real log file path is preserved."""
        path = make_config({"log_file": "/tmp/perch.log"})
        config = load_config(config_path=path)
        assert config["log_file"] == "/tmp/perch.log"


# ---------------------------------------------------------------------------
# classify normalization
# ---------------------------------------------------------------------------

class TestClassifyNormalization:

    def test_classify_true_defaults_to_csv(self, make_config):
        """classify: true defaults to 'csv'."""
        path = make_config({"classify": True})
        config = load_config(config_path=path)
        assert "csv" in config["classify"]

    def test_classify_parquet_validates(self, make_config):
        """classify: parquet validates and returns set."""
        path = make_config({"classify": "parquet"})
        config = load_config(config_path=path)
        assert "parquet" in config["classify"]


# ---------------------------------------------------------------------------
# parse_list_values edge cases
# ---------------------------------------------------------------------------

class TestParseListEdgeCases:

    def test_empty_and_whitespace(self):
        """Empty string returns {''}, whitespace-only returns {''}."""
        result = parse_list_values("")
        assert result == {""}

        result = parse_list_values("   ")
        assert result == {""}


# ---------------------------------------------------------------------------
# normalize_bool_string edge cases
# ---------------------------------------------------------------------------

class TestNormalizeBoolStringEdgeCases:
    """Non-string, non-bool types pass through."""

    def test_integer_zero(self):
        result = normalize_bool_string(0)
        assert result == 0

    def test_integer_one(self):
        result = normalize_bool_string(1)
        assert result == 1

    def test_list_passthrough(self):
        result = normalize_bool_string(["parquet"])
        assert result == ["parquet"]


# ---------------------------------------------------------------------------
# validate_embed_config edge cases
# ---------------------------------------------------------------------------

class TestValidateEmbedEdgeCases:

    def test_empty_string_raises(self):
        """Empty string raises ValueError ('' is not a valid filetype)."""
        with pytest.raises(ValueError):
            validate_embed_config("", {"serialized"})

    def test_hoplite_filetype(self):
        """hoplite is a valid filetype."""
        result = validate_embed_config("hoplite", {"serialized"})
        assert len(result) == 1
        assert result[0].filetype == "hoplite"

    def test_hoplite_with_columns(self):
        """hoplite + columns is valid."""
        result = validate_embed_config("hoplite-columns", {"serialized"})
        assert result[0].filetype == "hoplite"
        assert result[0].table_format == "columns"


# ---------------------------------------------------------------------------
# load_config path handling
# ---------------------------------------------------------------------------

class TestLoadConfigPaths:

    def test_source_output_are_path_objects(self, make_config):
        """source and output are converted to Path objects."""
        path = make_config()
        config = load_config(config_path=path)
        assert isinstance(config["source"], Path)
        assert isinstance(config["output"], Path)

    def test_config_path_is_directory_raises(self, tmp_path):
        """Passing a directory as config_path raises error."""
        with pytest.raises(Exception):
            load_config(config_path=str(tmp_path))
