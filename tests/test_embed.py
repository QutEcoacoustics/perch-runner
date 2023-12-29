
import os
import shutil
from pathlib import Path

import pandas as pd

from src import embed_audio_slim
from ml_collections import config_dict

def test_embed_one_file():

    embeddings = embed_audio_slim.embed_one_file("tests/files/100sec.wav")

    destination = "tests/output/100sec_embeddings.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3


def test_embed_files():

    # set up input folder

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/100sec.wav", one)
    shutil.copy("tests/files/100sec.wav", two)

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