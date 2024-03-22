
import os
import shutil
from pathlib import Path

import pandas as pd

from pytest_mock import mocker 

from src import embed_audio_slim

from ml_collections import config_dict

from src import config


default_embed_config = config.load_config(config.default_configs['generate'])


def test_embed_one_file():
    """
    tests both the embedding and the saving of the embeddings to csv
    """

    embeddings = embed_audio_slim.embed_one_file("tests/files/audio/100sec.wav", default_embed_config)

    destination = "tests/output/100sec_embeddings.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3


def test_embed_file_in_file_out():
    
    source = "tests/files/audio/100sec.wav"
    destination = "tests/output/100sec_embeddings.parquet"

    embed_audio_slim.embed_file_and_save(source, destination, default_embed_config)

    assert os.path.exists(destination)


def test_embed_file_in_folder_out():
    
    source = "tests/files/audio/100sec.wav"
    destination = "tests/output/"

    embed_audio_slim.embed_file_and_save(source, destination, default_embed_config)

    assert os.path.exists(destination + "100sec.parquet")


def test_embed_file_in_folder_out_file_exists(mocker):

    source = "tests/files/audio/100sec.wav"
    destination = "tests/output/"
    # create a file with the same name as the output file
    Path(destination + "100sec.parquet").touch()
    mocked_embed_one_file = mocker.patch('src.embed_audio_slim.embed_one_file')
    mocked_save_embeddings = mocker.patch('src.embed_audio_slim.save_embeddings')
    # skip if file exists is probably the default, but in case that changes
    my_config = config.merge_configs(default_embed_config, {'skip_if_file_exists': True})
    embed_audio_slim.embed_file_and_save(source, destination, my_config)
    assert os.path.exists(destination + "100sec.parquet")
    mocked_embed_one_file.assert_not_called()
    mocked_save_embeddings.assert_not_called()


def test_embed_file_in_folder_out_file_exists_false(mocker):

    source = "tests/files/audio/100sec.wav"
    destination = "tests/output/"
    # create a file with the same name as the output file
    Path(destination + "100sec.parquet").touch()
    mocked_embed_one_file = mocker.patch('src.embed_audio_slim.embed_one_file')
    mocked_save_embeddings = mocker.patch('src.embed_audio_slim.save_embeddings')
    # skip if file exists is probably the default, but in case that changes
    my_config = config.merge_configs(default_embed_config, {'skip_if_file_exists': False})
    embed_audio_slim.embed_file_and_save(source, destination, my_config)
    mocked_embed_one_file.assert_called()
    mocked_save_embeddings.assert_called()


def test_embed_folder():

    # set up input folder

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/audio/100sec.wav", one)
    shutil.copy("tests/files/audio/100sec.wav", two)

    source_folder = "tests/input/files"
    output_folder = "tests/output"

    embed_audio_slim.embed_folder(source_folder, output_folder, default_embed_config)

    expected_files = [Path(output_folder) / Path("one/100sec.parquet"), 
                      Path(output_folder) / Path("two/100sec.parquet")]

    assert os.path.exists(expected_files[0])
    assert os.path.exists(expected_files[0])
    
    # now read the embeddings back and check the shape and source columns are as expected
    df = pd.read_parquet(expected_files[0])
    assert df.shape == (20, 1283)
    assert df.source[0] == "one/100sec.wav"


