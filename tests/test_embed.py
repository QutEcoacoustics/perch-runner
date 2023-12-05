
import os

from src import embed_audio_slim
from ml_collections import config_dict

def test_embed_one_file():

    embed_audio_slim.embed_one_file("tests/files/100sec.wav", "tests/output")

    assert os.path.exists("tests/output/embeddings_0.csv")

