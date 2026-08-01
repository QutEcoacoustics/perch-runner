import argparse

import pytest

from src.config import load_config


@pytest.fixture
def io_dirs_fixture(tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return source, output


def test_perch_8_species_list_uses_final_labels(io_dirs_fixture):
    source, output = io_dirs_fixture
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        classify=True,
        model_choice="perch_8",
        perch_species_list="Ninox boobook, passer hemileucus",
        config_file=None,
    )

    config = load_config(None, args)
    assert config["perch_species_list"] == ["ninox boobook", "passer hemileucus"]


def test_perch_8_species_list_rejects_intermediate_ebird2021_labels(io_dirs_fixture):
    source, output = io_dirs_fixture
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        classify=True,
        model_choice="perch_8",
        perch_species_list="bknsti2",
        config_file=None,
    )

    with pytest.raises(ValueError, match="final label set for model_choice=perch_8"):
        load_config(None, args)


def test_perch_8_species_list_rejects_unknown_entries(io_dirs_fixture):
    source, output = io_dirs_fixture
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        classify=True,
        model_choice="perch_8",
        perch_species_list="not_a_real_species_name",
        config_file=None,
    )

    with pytest.raises(ValueError, match="final label set for model_choice=perch_8"):
        load_config(None, args)


def test_perch_v2_species_list_is_case_insensitive_and_canonicalized(io_dirs_fixture):
    source, output = io_dirs_fixture
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        classify=True,
        model_choice="perch_v2",
        perch_species_list="abavorana luctuosa, ABAVORANA LUCTUOSA, abeillia abeillei",
        config_file=None,
    )

    config = load_config(None, args)
    assert config["perch_species_list"] == ["Abavorana luctuosa", "Abeillia abeillei"]


def test_perch_v2_species_list_rejects_unknown_final_labels(io_dirs_fixture):
    source, output = io_dirs_fixture
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        classify=True,
        model_choice="perch_v2",
        perch_species_list="not_a_real_species_name",
        config_file=None,
    )

    with pytest.raises(ValueError, match="final label set for model_choice=perch_v2"):
        load_config(None, args)