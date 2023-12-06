
import os

from src import embed_audio_slim
from ml_collections import config_dict

def test_embed_one_file():

    embeddings = embed_audio_slim.embed_one_file("tests/files/100sec.wav")

    destination = "tests/output/100sec_embeddings.csv"
    embed_audio_slim.save_embeddings(embeddings, destination)

    assert os.path.exists(destination)
    assert len(embeddings.shape) == 3

