import re

import pytest

from src.sourcemap import (
    SourcemapConfig,
    build_sourcemap,
    compile_source_pattern,
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


class TestPresetSourcemap:
    def test_no_preset_returns_none(self):
        assert SourcemapConfig.from_inputs(sourcemap=None, sourcemap_values=None) is None
        assert build_sourcemap(None) is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown sourcemap"):
            SourcemapConfig.from_inputs(sourcemap="not_a_real_preset", sourcemap_values={"domain": "https://api.ecosounds.org"})

    def test_canonical_name_preset_maps_to_url(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap="canonical_to_ecosounds_original",
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)

        result = fn("site_0277/20210428T100000Z_Five-Rivers-Dry-A_909057.flac")
        assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_canonical_name_preset_with_timezone_offset(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap="canonical_to_ecosounds_original",
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)

        result = fn("site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav")
        assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_preset_no_match_returns_original(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap="canonical_to_ecosounds_original",
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)

        # not a canonical filename — no timestamp prefix
        result = fn("site_0277/not_a_canonical_name.flac")
        assert result == "site_0277/not_a_canonical_name.flac"

    def test_missing_required_token_raises(self):
        with pytest.raises(ValueError, match="missing token values"):
            SourcemapConfig.from_inputs(sourcemap="canonical_to_baw_original", sourcemap_values={})


class TestPresetCoverage:
    def test_canonical_to_baw_original(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap="canonical_to_baw_original",
            sourcemap_values={"domain": "https://api.acousticsobservatory.org"},
        )
        fn = build_sourcemap(cfg)
        assert fn("x/20210428T100000Z_Site_909057.wav") == "https://api.acousticsobservatory.org/audio_recordings/909057/original"

    def test_canonical_to_ecosounds_original(self):
        cfg = SourcemapConfig.from_inputs(sourcemap="canonical_to_ecosounds_original")
        fn = build_sourcemap(cfg)
        assert fn("x/20210428T100000Z_Site_909057.wav") == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_canonical_to_a2o_original(self):
        cfg = SourcemapConfig.from_inputs(sourcemap="canonical_to_a2o_original")
        fn = build_sourcemap(cfg)
        assert fn("x/20210428T100000Z_Site_909057.wav") == "https://api.acousticsobservatory.org/audio_recordings/909057/original"

    def test_baw_original(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap="baw_original",
            sourcemap_values={"domain": "https://api.acousticsobservatory.org", "arid": 909057},
        )
        fn = build_sourcemap(cfg)
        assert fn("x/any.wav") == "https://api.acousticsobservatory.org/audio_recordings/909057/original"

    def test_ecosounds_original(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap="ecosounds_original",
            sourcemap_values={"arid": 909057},
        )
        fn = build_sourcemap(cfg)
        assert fn("x/any.wav") == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_a2o_original(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap="a2o_original",
            sourcemap_values={"arid": 909057},
        )
        fn = build_sourcemap(cfg)
        assert fn("x/any.wav") == "https://api.acousticsobservatory.org.au/audio_recordings/909057/original"


class TestUnifiedSourcemap:
    def test_no_sourcemap_config_returns_none(self):
        assert build_sourcemap(None) is None

    def test_template_only_constant_mapping(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap_template="https://api.ecosounds.org/audio_recordings/1234/original"
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)
        assert fn("any/file.wav") == "https://api.ecosounds.org/audio_recordings/1234/original"

    def test_template_plus_pattern_preset(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap_template="https://api.ecosounds.org/audio_recordings/{arid}/original",
            sourcemap_pattern="canonical_filename",
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)
        result = fn("site_0277/20210428T100000Z_Five-Rivers-Dry-A_909057.flac")
        assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    def test_template_plus_custom_pattern_and_values(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap_template="{domain}/audio_recordings/{arid}/original",
            sourcemap_pattern=r"^(?P<arid>\d+)\.wav$",
            sourcemap_values={"domain": "https://api.ecosounds.org"},
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)
        assert fn("1234.wav") == "https://api.ecosounds.org/audio_recordings/1234/original"

    def test_pattern_no_match_returns_original(self):
        sourcemap_config = SourcemapConfig.from_inputs(
            sourcemap_template="https://api.ecosounds.org/audio_recordings/{arid}/original",
            sourcemap_pattern="canonical_filename",
        )
        fn = build_sourcemap(sourcemap_config)
        assert callable(fn)
        assert fn("site_0277/not_a_canonical_name.flac") == "site_0277/not_a_canonical_name.flac"

    def test_unknown_sourcemap_raises(self):
        with pytest.raises(ValueError, match="Unknown sourcemap"):
            SourcemapConfig.from_inputs(sourcemap="not_a_real_preset")
