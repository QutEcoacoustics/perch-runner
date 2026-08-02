import pytest

from src.sourcemap import SourcemapConfig, build_extra_columns_map, build_sourcemap


class TestSourcemapConfig:
    def test_no_inputs_returns_none(self):
        assert SourcemapConfig.from_inputs() is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown sourcemap name"):
            SourcemapConfig.from_inputs(sourcemap_name="not_a_real_preset")

    def test_empty_template_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            SourcemapConfig.from_inputs(sourcemap_template="   ")

    def test_invalid_pattern_rejected(self):
        with pytest.raises(ValueError, match="Invalid source_map_pattern"):
            SourcemapConfig.from_inputs(
                sourcemap_template="{audio_recording_id}",
                file_metadata_pattern=r"(unclosed",
            )

    def test_missing_template_tokens_rejected(self):
        with pytest.raises(ValueError, match="missing token values"):
            SourcemapConfig.from_inputs(
                sourcemap_template="{domain}/audio_recordings/{audio_recording_id}/original",
                file_metadata={},
            )


class TestSourcemapMapping:
    def test_none_config_returns_identity_mapper(self):
        fn = build_sourcemap(None)
        assert callable(fn)
        assert fn("x/file.wav") == "x/file.wav"

    def test_named_template_with_static_tokens(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap_name="baw_original",
            file_metadata={"domain": "https://api.acousticsobservatory.org", "audio_recording_id": 909057},
        )
        fn = build_sourcemap(cfg)
        assert fn("x/any.wav") == "https://api.acousticsobservatory.org/audio_recordings/909057/original"

    def test_named_template_with_pattern_extracted_audio_recording_id(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap_name="ecosounds_original",
            file_metadata_pattern="canonical_filename",
        )
        fn = build_sourcemap(cfg)
        assert fn("site_0277/20210428T100000Z_Five-Rivers-Dry-A_909057.flac") == (
            "https://api.ecosounds.org/audio_recordings/909057/original"
        )

    def test_pattern_no_match_raises_for_missing_tokens(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap_name="ecosounds_original",
            file_metadata_pattern="canonical_filename",
        )
        fn = build_sourcemap(cfg)
        with pytest.raises(ValueError, match="Missing sourcemap token value"):
            fn("site_0277/not_a_canonical_name.flac")

    def test_custom_template_pattern_and_static_token(self):
        cfg = SourcemapConfig.from_inputs(
            sourcemap_template="{domain}/audio_recordings/{audio_recording_id}/original",
            file_metadata_pattern=r"^(?P<audio_recording_id>\d+)\.wav$",
            file_metadata={"domain": "https://api.ecosounds.org"},
        )
        fn = build_sourcemap(cfg)
        assert fn("1234.wav") == "https://api.ecosounds.org/audio_recordings/1234/original"


class TestExtraColumnsMap:
    def test_none_config_returns_empty_dict_mapper(self):
        fn = build_extra_columns_map(None, ["audio_recording_id"])
        assert fn("x/file.wav") == {}

    def test_extra_columns_from_pattern_and_static_tokens(self):
        cfg = SourcemapConfig.from_inputs(
            file_metadata_pattern="canonical_filename",
            file_metadata={"domain": "https://api.ecosounds.org"},
        )
        fn = build_extra_columns_map(cfg, ["audio_recording_id", "domain", "missing"])
        assert fn("x/20210428T100000Z_Five-Rivers-Dry-A_909057.flac") == {
            "audio_recording_id": "909057",
            "domain": "https://api.ecosounds.org",
        }
