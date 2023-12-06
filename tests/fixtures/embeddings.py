import numpy as np
import pytest


@pytest.fixture(scope="session")
def random_embeddings_01() -> np.array:
    # a random np array with 3 channels, 123 frames
    num_frames = 12
    embeddings = np.random.rand(num_frames, 3, 1280).astype(np.float32)
    hop_size = 5
    offsets = np.repeat(np.arange(0,embeddings.shape[0]*hop_size,hop_size, dtype=np.float32), (3)).reshape(embeddings.shape[0],3,1)
    return np.concatenate((offsets, embeddings), axis=2)


   

