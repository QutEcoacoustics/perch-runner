"""Unit tests for app module-level initialization and model-choice resolution."""

import os
from types import SimpleNamespace

import pytest

# Import app module to trigger TF_CPP_MIN_LOG_LEVEL setup
from src import app  # noqa: F401


class TestModuleLevel:

    def test_tf_cpp_log_level_set_at_import(self):
        """TF_CPP_MIN_LOG_LEVEL is set when app module is imported."""
        # The import at the top of this file already triggers the setdefault.
        # Verify it's set (setdefault won't overwrite if already present).
        assert "TF_CPP_MIN_LOG_LEVEL" in os.environ
        # Value should be '1' (the default) or whatever was already set
        assert os.environ["TF_CPP_MIN_LOG_LEVEL"] in ("1", "2", "3")


class TestRecognizerModelChoiceResolution:

    def test_infers_model_choice_from_embedding_dim(self):
        config = {
            "model_choice": "perch_v2",
            "recognizers": [{"classifier": {"classes": ["owl"]}}],
        }
        classifier_config_list = SimpleNamespace(
            embedding_model_name=None,
            embedding_dim=(1280,),
        )

        with pytest.warns(UserWarning, match="Embedding model name not provided"):
            resolved = app._resolve_model_choice_for_recognizers(
                config,
                model_choice_explicit=False,
                classifier_config_list=classifier_config_list,
            )

        assert resolved["model_choice"] == "perch_8"

    def test_raises_when_embedding_dim_is_ambiguous(self, monkeypatch):
        config = {
            "model_choice": "perch_v2",
            "recognizers": [{"classifier": {"classes": ["owl"]}}],
        }
        classifier_config_list = SimpleNamespace(
            embedding_model_name=None,
            embedding_dim=(1280,),
        )
        monkeypatch.setattr(
            app,
            "MODELS",
            {
                "a": {"embedding_dim": 1280},
                "b": {"embedding_dim": 1280},
            },
        )

        with pytest.raises(ValueError, match="embedding model name not provided, and can't be determined from embedding dimension"):
            app._resolve_model_choice_for_recognizers(
                config,
                model_choice_explicit=False,
                classifier_config_list=classifier_config_list,
            )
