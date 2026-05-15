"""
Integration tests that exercise the full CLI pipeline.

When run inside a container: calls `python src/app.py` as a subprocess.
When run on the host: calls `docker run ...` to exercise the container.

The `runner` fixture in conftest.py auto-detects the environment.

Run with: pytest tests/integration/
"""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import data_frames


FIXTURES_DIR = Path("tests/files")
A2O_FLAC = "20220502T075930+1000_Minjerribah-Dry-B_1088507.flac"


class TestEmbedCLI:

    def test_embed_nested_wav(self, runner, workspace):
        """Basic: wav in a subdirectory, default --embed."""
        source, output, _ = workspace
        site = source / "deploy1"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

        runner(source, output, "--embed")

        parquet = output / "embeddings" / "deploy1" / "100sec.wav" / "embeddings.parquet"
        assert parquet.exists()
        df = pd.read_parquet(parquet)
        assert len(df) >= 19
        assert "embeddings" in df.columns
        assert df["source"].iloc[0] == "deploy1/100sec.wav"

    def test_embed_flat_source(self, runner, workspace):
        """Audio at root level, no subdirectory."""
        source, output, _ = workspace
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", source)

        runner(source, output, "--embed")

        parquet = output / "embeddings" / "100sec.wav" / "embeddings.parquet"
        assert parquet.exists()
        df = pd.read_parquet(parquet)
        assert len(df) >= 19
        assert df["source"].iloc[0] == "100sec.wav"

    def test_embed_a2o_flac(self, runner, workspace):
        """A2O-style FLAC in a site subdirectory."""
        source, output, _ = workspace
        site = source / "Minjerribah-Dry-B"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / A2O_FLAC, site)

        runner(source, output, "--embed")

        parquet = output / "embeddings" / "Minjerribah-Dry-B" / A2O_FLAC / "embeddings.parquet"
        assert parquet.exists()
        df = pd.read_parquet(parquet)
        assert len(df) >= 6
        assert df["source"].iloc[0] == f"Minjerribah-Dry-B/{A2O_FLAC}"

    def test_embed_columns_format(self, runner, workspace):
        """--embed parquet --embedding_table_format columns produces column-per-dimension."""
        source, output, _ = workspace
        site = source / "site"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

        runner(source, output, "--embed", "parquet", "--embedding_table_format", "columns")

        parquet = output / "embeddings" / "site" / "100sec.wav" / "embeddings.parquet"
        assert parquet.exists()
        df = pd.read_parquet(parquet)
        assert "f0000" in df.columns
        assert "embeddings" not in df.columns

    def test_embed_with_config_file(self, runner, workspace):
        """Config file specifies embed format."""
        source, output, config_dir = workspace
        site = source / "site"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

        config_file = config_dir / "config.yml"
        config_file.write_text(
            f"source: {source}\noutput: {output}\nembed: parquet\n"
        )

        runner(source, output, config_file=config_file)

        parquet = output / "embeddings" / "site" / "100sec.wav" / "embeddings.parquet"
        assert parquet.exists()

    def test_embed_with_file_glob(self, runner, workspace):
        """Explicit --file_glob overrides auto-detection."""
        source, output, _ = workspace
        # Two levels deep
        deep = source / "site" / "date"
        deep.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", deep)

        runner(source, output, "--embed", "--file_glob", "*/*/*")

        parquet = output / "embeddings" / "site" / "date" / "100sec.wav" / "embeddings.parquet"
        assert parquet.exists()

    def test_no_embed_flag_errors(self, runner, workspace):
        """Without --embed/--classify, CLI should fail validation."""
        source, output, _ = workspace
        site = source / "site"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

        with pytest.raises(RuntimeError, match="At least one of --embed or --classify"):
            runner(source, output)

        assert not (output / "embeddings").exists()
        assert not (output / "hoplite").exists()

    def test_embed_writes_db_at_default_path(self, runner, workspace):
        """Embedding writes the database to output/db by default."""
        source, output, _ = workspace
        site = source / "site"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

        runner(source, output, "--embed", "parquet")

        assert (output / "db").exists()
        assert (output / "embeddings").exists()


# Import model metadata from source of truth
from src.version import MODELS


class TestEmbedCLIModels:
    """Integration tests parametrized over all cached models."""

    @pytest.mark.parametrize("model_choice", MODELS.keys(), ids=MODELS.keys())
    def test_embed_model(self, runner, workspace, model_choice):
        """Embed with each model, verify parquet output."""
        source, output, _ = workspace
        site = source / "site"
        site.mkdir()
        shutil.copy(FIXTURES_DIR / "audio" / "100sec.wav", site)

        runner(source, output, "--embed", "--model_choice", model_choice)

        parquet = output / "embeddings" / "site" / "100sec.wav" / "embeddings.parquet"
        assert parquet.exists()
        df = pd.read_parquet(parquet)
        assert len(df) >= 19
        assert "embeddings" in df.columns
        assert df["source"].iloc[0] == "site/100sec.wav"
