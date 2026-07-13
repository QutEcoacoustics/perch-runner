import argparse
import json
from pathlib import Path

import pytest
import yaml

from src.config import (
    config_to_json,
    load_config,
    normalize_bool_string,
    parse_list_values,
    validate_single_value,
)
from src.sourcemap import SourcemapConfig


@pytest.fixture
def io_dirs(tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return source, output


def _write_config(path: Path, payload: dict):
    if path.suffix == ".json":
        path.write_text(json.dumps(payload))
    else:
        path.write_text(yaml.safe_dump(payload))


class TestHelpers:
    def test_parse_list_values(self):
        assert parse_list_values("Parquet, csv") == ["csv", "parquet"]

    def test_normalize_bool_string_mutates_dict(self):
        cfg = {"embed": "true", "save_db": "false"}
        normalize_bool_string(cfg, "embed")
        normalize_bool_string(cfg, "save_db")
        assert cfg["embed"] is True
        assert cfg["save_db"] is False

    def test_normalize_bool_string_missing_key_noop(self):
        cfg = {"x": 1}
        normalize_bool_string(cfg, "embed")
        assert cfg == {"x": 1}

    def test_validate_single_value_missing_key_noop(self):
        cfg = {}
        assert validate_single_value(cfg, "model_choice") is None

    def test_validate_single_value_rejects_invalid(self):
        with pytest.raises(ValueError, match="Invalid model_choice"):
            validate_single_value({"model_choice": "bad_model"}, "model_choice")

    def test_config_to_json_handles_paths(self, tmp_path):
        rendered = config_to_json({"source": tmp_path / "in", "flags": {"a", "b"}})
        parsed = json.loads(rendered)
        assert parsed["source"].endswith("/in")
        assert parsed["flags"] == ["a", "b"]

    def test_config_to_json_handles_sourcemap_config(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap="canonical_to_baw_original",
            sourcemap_values={"domain": "https://api.ecosounds.org"},
        )
        rendered = config_to_json({"sourcemap_config": sourcemap_config})
        parsed = json.loads(rendered)
        assert parsed["sourcemap_config"]["sourcemap_template"] == "{domain}/audio_recordings/{arid}/original"
        assert parsed["sourcemap_config"]["sourcemap_values"] == {"domain": "https://api.ecosounds.org"}


class TestLoadConfig:
    def test_load_config_from_yaml_and_defaults(self, io_dirs, tmp_path):
        source, output = io_dirs
        config_file = tmp_path / "config.yml"
        _write_config(
            config_file,
            {
                "source": str(source),
                "output": str(output),
                "embed": True,
            },
        )

        config = load_config(str(config_file), None)

        assert config["source"] == source
        assert config["output"] == output
        assert config["embed"] is True
        assert config["embeddings_table_format"] == "serialized"
        assert config["embeddings_table_filetype"] == "parquet"
        assert config["embeddings_output_path_template"] == "{analysis}{ext}"

    def test_cli_args_override_file(self, io_dirs, tmp_path):
        source, output = io_dirs
        config_file = tmp_path / "config.yml"
        _write_config(
            config_file,
            {
                "source": str(source),
                "output": str(output),
                "embed": True,
                "model_choice": "perch_v2",
            },
        )

        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            model_choice="perch_8",
            config_file=None,
        )

        config = load_config(str(config_file), args)
        assert config["model_choice"] == "perch_8"

    def test_recognizers_only_is_valid_action(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            recognizers="tests/files/configs/koala.json",
            config_file=None,
        )

        config = load_config(None, args)

        assert config["recognizers"]
        assert config["model_choice"] == "perch_8"

    def test_conflicting_classify_false_and_classify_filetype_raises(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=False,
            classify_filetype="csv",
            config_file=None,
        )

        with pytest.raises(ValueError, match="Cannot specify --classify false"):
            load_config(None, args)

    def test_requires_at_least_one_action(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(source=str(source), output=str(output), config_file=None)

        with pytest.raises(ValueError, match="At least one of --embed, --classify, --save_db or --recognizers"):
            load_config(None, args)

    def test_embed_false_with_embed_related_keys_warns_and_disables_embed(self, io_dirs, tmp_path):
        source, output = io_dirs
        config_file = tmp_path / "config.yml"
        _write_config(
            config_file,
            {
                "source": str(source),
                "output": str(output),
                "embed": False,
                "embeddings_table_format": "columns",
                "save_db": True,
            },
        )

        with pytest.warns(UserWarning, match="embed is explicitly set to false"):
            config = load_config(str(config_file), None)

        assert config["embed"] is False

    def test_sourcemap_values_json_string_from_cli(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            sourcemap="canonical_to_baw_original",
            sourcemap_values='{"domain": "https://api.ecosounds.org"}',
            config_file=None,
        )

        config = load_config(None, args)
        assert config["sourcemap"] == "canonical_to_baw_original"
        assert config["sourcemap_values"] == {"domain": "https://api.ecosounds.org"}
        assert isinstance(config["sourcemap_config"], SourcemapConfig)

    def test_sourcemap_values_without_source_raises(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            sourcemap_values={"domain": "https://api.ecosounds.org"},
            config_file=None,
        )

        with pytest.raises(ValueError, match="sourcemap_values requires sourcemap or sourcemap_template"):
            load_config(None, args)

    def test_invalid_sourcemap_raises(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            sourcemap="not_real",
            config_file=None,
        )

        with pytest.raises(ValueError, match="Invalid sourcemap value"):
            load_config(None, args)
