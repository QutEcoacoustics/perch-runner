
import os
import shutil
from pathlib import Path

import pandas as pd

from src import embed_audio_slim

from ml_collections import config_dict


def test_embed_one_file():
    """
    tests both the embedding and the saving of the embeddings to csv
    """

    embeddings = embed_audio_slim.embed_one_file("tests/files/audio/100sec.wav")

    destination = "tests/output/100sec_embeddings.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3


def test_embed_flac():
    """
    tests both the embedding and the saving of the embeddings to csv
    """

    embeddings = embed_audio_slim.embed_one_file("tests/files/audio/segment.flac")

    destination = "tests/output/segment.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3


def test_embed_one_file_empty_config():
    """
    tests both the embedding and the saving of the embeddings to csv
    """

    embeddings = embed_audio_slim.embed_one_file("tests/files/audio/100sec.wav", config=config_dict.create(**{}))

    destination = "tests/output/100sec_embeddings.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3



def test_embed_file_in_file_out():
    """
    Tests embedding and saving to a specified parquet filename
    Also tests that the canonical filename is parsed correctly
    """

    original_source = "tests/files/audio/100sec.wav"
    source = "tests/input/20240101T123456Z_my-amazing-site_2468.wav"
    shutil.copy(original_source, source)
    destination = "tests/output/100sec_embeddings.parquet"

    embed_audio_slim.embed_file_and_save(source, destination)

    assert os.path.exists(destination)

    # now read the embeddings back and check the shape and source columns are as expected
    df = pd.read_parquet(destination)
    assert df.shape == (20, 1283)
    assert df.source[0] == "https://api.ecosounds.org/audio_recordings/2468/original"


def test_embed_file_in_folder_out():
    
    source = "tests/files/audio/100sec.wav"
    destination = "tests/output/"

    embed_audio_slim.embed_file_and_save(source, destination)

    assert os.path.exists(destination + "100sec.parquet")


def test_find_audio_files():

    source_dir = Path("tests/input/files/")


    files = [
        (Path("one/a.wav"), True),
        (Path("one/b.FLAC"), True),
        (Path("two/b.flac"), True),
        (Path("two/b.mp3"), True),
        (Path("three/b.txt"), False),
        (Path("three/b.ogg"), True),
        (Path("four/b.json"), False),
        (Path("four/b.WAV"), True)
    ]

    for file in files:
        file = source_dir / file[0]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()

    expected_output = [file[0] for file in files if file[1]]

    audio_files = embed_audio_slim.get_audio_files("tests/input/files")

    assert sorted(audio_files) == sorted(expected_output)




def test_embed_folder():

    # set up input folder

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/audio/100sec.wav", one)
    shutil.copy("tests/files/audio/100sec.wav", two)
    shutil.copy("tests/files/audio/segment.flac", two)

    source_folder = "tests/input/files"
    output_folder = "tests/output"

    embed_audio_slim.embed_folder(source_folder, output_folder)

    expected_files = [Path(output_folder) / Path("one/100sec.parquet"), 
                      Path(output_folder) / Path("two/100sec.parquet"),
                      Path(output_folder) / Path("two/segment.parquet"),]

    assert os.path.exists(expected_files[0])
    assert os.path.exists(expected_files[1])
    assert os.path.exists(expected_files[2])
    
    # now read the embeddings back and check the shape and source columns are as expected
    df = pd.read_parquet(expected_files[0])
    assert df.shape == (20, 1283)
    assert df.source[0] == "one/100sec.wav"

    df = pd.read_parquet(expected_files[2])
    # this test file has duration 00:01:46.905 = 21x5s segments
    assert df.shape == (21, 1283)
    assert df.source[2] == "two/segment.flac"


def test_embed_one_file_bit_depth():
    """
    Tests that the bit depth specified in the config is correctly applied to the embeddings array.
    """
    # Test each supported bit depth
    for bit_depth in [16, 32, 64]:
        config = config_dict.create(
            bit_depth=bit_depth
        )
        
        embeddings = embed_audio_slim.embed_one_file("tests/files/audio/100sec.wav", config)
        
        # Check that the dtype of the embeddings matches what we expect from the config
        expected_dtype = embed_audio_slim.DTYPE_MAPPING[bit_depth]
        assert embeddings.dtype == expected_dtype, f"Expected dtype {expected_dtype} for bit_depth {bit_depth}, but got {embeddings.dtype}"
        
        # Basic shape checks to ensure the embeddings are still valid
        assert len(embeddings.shape) == 3