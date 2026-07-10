import re

import pytest

from src.sourcemap import (
    apply_source_map,
    build_sourcemap_from_preset,
    compile_source_pattern,
    create_sourcemap_function,
)


class TestCompileSourcePattern:

    def test_valid_pattern(self):
        pattern = compile_source_pattern(r"(\d+)")
        assert isinstance(pattern, re.Pattern)

    def test_invalid_pattern(self):
        with pytest.raises(ValueError, match="Invalid source_map_pattern"):
            compile_source_pattern(r"(unclosed")

    def test_complex_pattern(self):
        pattern = compile_source_pattern(r"(\d{8}T\d{6}[Z+-]\d{0,4})_(.+?)_(\d+)\.\w+")
        assert isinstance(pattern, re.Pattern)


class TestApplySourceMap:

    def test_simple_group_replacement(self):
        pattern = compile_source_pattern(r"(.+)_(\d+)\.wav")
        result = apply_source_map("site_a/20210428_12345.wav", pattern, "recordings/{2}")
        assert result == "recordings/12345"

    def test_full_match_group_zero(self):
        pattern = compile_source_pattern(r"\d+")
        result = apply_source_map("folder/file_42.wav", pattern, "id_{0}")
        assert result == "id_42"

    def test_multiple_groups(self):
        pattern = compile_source_pattern(r"(\d{8}T\d{6}Z)_(.+?)_(\d+)\.flac")
        filename = "site_0277/20210428T100000Z_Five-Rivers-Dry-A_909057.flac"
        result = apply_source_map(filename, pattern, "https://api.example.org/recordings/{3}/original")
        assert result == "https://api.example.org/recordings/909057/original"

    def test_no_match_returns_original(self):
        pattern = compile_source_pattern(r"NOMATCH")
        result = apply_source_map("folder/myfile.wav", pattern, "replaced")
        assert result == "folder/myfile.wav"

    def test_operates_on_basename(self):
        """Pattern matches against basename only, not the full path."""
        pattern = compile_source_pattern(r"^(\w+)\.wav$")
        result = apply_source_map("deep/nested/path/myfile.wav", pattern, "{1}.parquet")
        assert result == "myfile.parquet"

    def test_unused_group_placeholder_preserved(self):
        """If a group is None (optional and unmatched), its placeholder stays."""
        pattern = compile_source_pattern(r"(\w+)(?:_(\d+))?\.wav")
        result = apply_source_map("myfile.wav", pattern, "{1}_{2}")
        # group 2 didn't match, so {2} stays
        assert result == "myfile_{2}"


class TestCreateSourcemapFunction:

    def test_returns_callable(self):
        fn = create_sourcemap_function(r"(\d+)", "id_{1}")
        assert callable(fn)

    def test_function_applies_pattern(self):
        fn = create_sourcemap_function(r"(.+)_(\d+)\.wav", "audio/{2}/{1}.parquet")
        result = fn("input/site-name_99999.wav")
        assert result == "audio/99999/site-name.parquet"

    def test_function_no_match(self):
        fn = create_sourcemap_function(r"NEVER_MATCH", "replaced")
        result = fn("some/path/file.wav")
        assert result == "some/path/file.wav"

    def test_invalid_pattern_raises_early(self):
        with pytest.raises(ValueError):
            create_sourcemap_function(r"(bad", "template")


class TestPresetSourcemap:
    def test_no_preset_returns_none(self):
        assert build_sourcemap_from_preset(None, None) is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown sourcemap_preset"):
            build_sourcemap_from_preset("not_a_real_preset", {"domain": "https://api.ecosounds.org"})

    def test_canonical_name_preset_maps_to_url(self):
        fn = build_sourcemap_from_preset(
            "canonical_name_to_original_recording_url",
            {"domain": "https://api.ecosounds.org"},
        )

        result = fn("site_0277/20210428T100000Z_Five-Rivers-Dry-A_909057.flac")
        assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_canonical_name_preset_with_timezone_offset(self):
        fn = build_sourcemap_from_preset(
            "canonical_name_to_original_recording_url",
            {"domain": "https://api.ecosounds.org"},
        )

        result = fn("site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav")
        assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_preset_no_match_returns_original(self):
        fn = build_sourcemap_from_preset(
            "canonical_name_to_original_recording_url",
            {"domain": "https://api.ecosounds.org"},
        )

        # not a canonical filename — no timestamp prefix
        result = fn("site_0277/not_a_canonical_name.flac")
        assert result == "site_0277/not_a_canonical_name.flac"

    def test_missing_required_token_raises(self):
        with pytest.raises(ValueError, match="missing token values"):
            build_sourcemap_from_preset("canonical_name_to_original_recording_url", {})
