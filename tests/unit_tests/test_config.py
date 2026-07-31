import argparse
import json
from pathlib import Path

import pytest
import yaml
import src.config as config_module

from src.config import (
    config_to_json,
    load_config,
    normalize_perch_species_list,
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

    def test_normalize_perch_species_list_inline_csv_and_newline(self):
        cfg = {"perch_species_list": "Koala,Emu\nCurrawong"}
        normalize_perch_species_list(cfg)
        assert cfg["perch_species_list"] == ["Koala", "Emu", "Currawong"]

    def test_normalize_perch_species_list_from_list(self):
        cfg = {"perch_species_list": ["Koala", "Emu", "Koala"]}
        normalize_perch_species_list(cfg)
        assert cfg["perch_species_list"] == ["Koala", "Emu"]

    def test_normalize_perch_species_list_from_file(self, tmp_path):
        species_file = tmp_path / "species.txt"
        species_file.write_text("Koala\nEmu,Currawong", encoding="utf-8")
        cfg = {"perch_species_list": str(species_file)}
        normalize_perch_species_list(cfg)
        assert cfg["perch_species_list"] == ["Koala", "Emu", "Currawong"]

    def test_normalize_perch_species_list_from_preset_key(self, tmp_path, monkeypatch):
        species_file = tmp_path / "preset_species.txt"
        species_file.write_text("Koala\nEmu,Currawong", encoding="utf-8")
        monkeypatch.setattr(
            config_module,
            "SPECIES_LIST_PRESETS",
            {"australian_birds_01": str(species_file)},
        )

        cfg = {"perch_species_list": "australian_birds_01"}
        normalize_perch_species_list(cfg)
        assert cfg["perch_species_list"] == ["Koala", "Emu", "Currawong"]

    def test_normalize_perch_species_list_prefers_existing_path_over_preset_key(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()

        path_named_like_preset = config_dir / "australian_birds_01"
        path_named_like_preset.write_text("Path Species", encoding="utf-8")

        preset_species_file = tmp_path / "preset_species.txt"
        preset_species_file.write_text("Preset Species", encoding="utf-8")
        monkeypatch.setattr(
            config_module,
            "SPECIES_LIST_PRESETS",
            {"australian_birds_01": str(preset_species_file)},
        )

        cfg = {"perch_species_list": "australian_birds_01"}
        normalize_perch_species_list(cfg, config_file_dir=config_dir)
        assert cfg["perch_species_list"] == ["Path Species"]

    def test_normalize_bool_string_mutates_dict(self):
        cfg = {"embed": "true", "save_db": "false"}
        normalize_bool_string(cfg, "embed")
        normalize_bool_string(cfg, "save_db")
        assert cfg["embed"] is True
        assert cfg["save_db"] is False

    def test_normalize_bool_string_leaves_empty_string_unchanged(self):
        cfg = {"embed": ""}
        normalize_bool_string(cfg, "embed")
        assert cfg["embed"] == ""

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
            sourcemap_name="baw_original",
            file_metadata={"domain": "https://api.ecosounds.org", "audio_recording_id": 1234},
        )
        rendered = config_to_json({"sourcemap_config": sourcemap_config})
        parsed = json.loads(rendered)
        assert parsed["sourcemap_config"]["sourcemap_template"] == "{domain}/audio_recordings/{audio_recording_id}/original"
        assert parsed["sourcemap_config"]["file_metadata"] == {
            "domain": "https://api.ecosounds.org",
            "audio_recording_id": "1234",
        }


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
        assert config["classify_output_path_template"] == "{analysis}{ext}"
        assert config["perch_max_detections_per_window"] == 10
        assert config["perch_species_list"] is None

    def test_perch_species_list_normalized_from_inline_string(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            perch_species_list="Phascolarctos cinereus,Cuculus canorus\nNinox boobook",
            config_file=None,
        )

        config = load_config(None, args)
        assert config["perch_species_list"] == [
            "Phascolarctos cinereus",
            "Cuculus canorus",
            "Ninox boobook",
        ]

    def test_perch_species_list_normalized_from_file_path(self, io_dirs, tmp_path):
        source, output = io_dirs
        species_file = tmp_path / "species.txt"
        species_file.write_text("Phascolarctos cinereus\nCuculus canorus,Ninox boobook", encoding="utf-8")

        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            perch_species_list=str(species_file),
            config_file=None,
        )

        config = load_config(None, args)
        assert config["perch_species_list"] == [
            "Phascolarctos cinereus",
            "Cuculus canorus",
            "Ninox boobook",
        ]

    def test_perch_max_detections_per_window_parsed_as_int(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            perch_max_detections_per_window="7",
            config_file=None,
        )

        config = load_config(None, args)
        assert config["perch_max_detections_per_window"] == 7

    def test_invalid_perch_max_detections_per_window_raises(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            perch_max_detections_per_window="not_an_int",
            config_file=None,
        )

        with pytest.raises(ValueError, match="perch_max_detections_per_window must be an integer"):
            load_config(None, args)

    def test_empty_perch_species_list_raises(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            perch_species_list=[],
            config_file=None,
        )

        with pytest.raises(ValueError, match="perch_species_list must contain at least one species"):
            load_config(None, args)

    def test_classify_does_not_require_species_list_by_default(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            config_file=None,
        )

        config = load_config(None, args)
        assert config["perch_species_list"] is None
        assert config["classify_require_species_list"] is False

    def test_classify_can_require_species_list_explicitly(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            classify_require_species_list=True,
            config_file=None,
        )

        with pytest.raises(ValueError, match="perch_species_list must be provided"):
            load_config(None, args)

    def test_classify_can_skip_species_list_when_not_required(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=True,
            classify_require_species_list=False,
            config_file=None,
        )

        config = load_config(None, args)
        assert config["perch_species_list"] is None
        assert config["classify_require_species_list"] is False

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

    def test_classify_false_with_classify_related_keys_warns_and_disables_classify(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            classify=False,
            classify_filetype="csv",
            save_db=True,
            config_file=None,
        )

        with pytest.warns(UserWarning, match="classify is explicitly disabled"):
            config = load_config(None, args)

        assert config["classify"] is False

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

        with pytest.warns(UserWarning, match="embed is explicitly disabled"):
            config = load_config(str(config_file), None)

        assert config["embed"] is False

    def test_embed_false_ignores_invalid_embeddings_template(self, io_dirs, tmp_path):
        source, output = io_dirs
        config_file = tmp_path / "config.yml"
        _write_config(
            config_file,
            {
                "source": str(source),
                "output": str(output),
                "embed": False,
                "embeddings_output_path_template": "{not_a_real_token}",
                "save_db": True,
            },
        )

        with pytest.warns(UserWarning, match="embed is explicitly disabled"):
            config = load_config(str(config_file), None)

        assert config["embed"] is False

    def test_recognizer_settings_without_recognizers_are_ignored(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            save_db=True,
            recognizer_output_path_template="{bad_token}",
            recognizer_output_path_type="flat",
            recognizer_results_filetype="csv",
            config_file=None,
        )

        with pytest.warns(UserWarning, match="recognizers is explicitly disabled"):
            config = load_config(None, args)

        assert not config["recognizers"]

    def test_file_metadata_json_string_from_cli(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            sourcemap_name="baw_original",
            file_metadata='{"domain": "https://api.ecosounds.org", "audio_recording_id": 1234}',
            config_file=None,
        )

        config = load_config(None, args)
        assert config["sourcemap_name"] == "baw_original"
        assert config["file_metadata"] == {"domain": "https://api.ecosounds.org", "audio_recording_id": 1234}
        assert isinstance(config["sourcemap_config"], SourcemapConfig)

    def test_file_metadata_without_sourcemap_is_allowed(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            file_metadata={"domain": "https://api.ecosounds.org"},
            config_file=None,
        )

        config = load_config(None, args)
        assert config["file_metadata"] == {"domain": "https://api.ecosounds.org"}
        assert isinstance(config["sourcemap_config"], SourcemapConfig)

    def test_invalid_sourcemap_raises(self, io_dirs):
        source, output = io_dirs
        args = argparse.Namespace(
            source=str(source),
            output=str(output),
            embed=True,
            sourcemap_name="not_real",
            config_file=None,
        )

        with pytest.raises(ValueError, match="Invalid sourcemap_name value"):
            load_config(None, args)
