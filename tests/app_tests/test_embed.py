
import os
import shutil
from pathlib import Path

import pandas as pd

from src import embed_audio_slim
from src import batch
from ml_collections import config_dict


def test_embed_one_file():

    embeddings = embed_audio_slim.embed_one_file("tests/files/audio/100sec.wav")

    destination = "tests/output/100sec_embeddings.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3


def test_embed_one_file_and_save():
    
    source = "tests/files/audio/100sec.wav"
    destination = "tests/output/100sec_embeddings.parquet"

    embed_audio_slim.embed_file_and_save(source, destination)

    assert os.path.exists(destination)


def test_embed_files():

    # set up input folder

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/audio/100sec.wav", one)
    shutil.copy("tests/files/audio/100sec.wav", two)

    source_folder = "tests/input/files"
    output_folder = "tests/output"

    embed_audio_slim.embed_files(source_folder, output_folder)

    expected_files = [Path(output_folder) / Path("one/100sec.parquet"), 
                      Path(output_folder) / Path("two/100sec.parquet")]

    assert os.path.exists(expected_files[0])
    assert os.path.exists(expected_files[0])
    
    # now read the embeddings back and check the shape and source columns are as expected
    df = pd.read_parquet(expected_files[0])
    assert df.shape == (20, 1283)
    assert df.source[0] == "one/100sec.wav"


def test_batch_entrypoint_item_0_1():
    """Test batch processing from row 0 to 1 of the batch list"""

    batch.batch('generate', source_csv='tests/files/batch_lists/batch_embed.csv', start_row=0, end_row=1, config_file=None)

    assert Path('tests/output/100sec.parquet').exists()
    assert Path('tests/output/some_subfolder/200sec.parquet').exists()
    assert not Path('tests/output/some_subfolder/100sec_again.parquet').exists()


def test_batch_entrypoint_item_2():
    """Test batch processing from row 2 to 2 of the batch list"""

    batch.batch('generate', source_csv='tests/files/batch_lists/batch_embed.csv', start_row=2, end_row=2, config_file=None)

    assert not Path('tests/output/100sec.parquet').exists()
    assert not Path('tests/output/some_subfolder/200sec.parquet').exists()
    assert Path('tests/output/some_subfolder/100sec_again.parquet').exists()
