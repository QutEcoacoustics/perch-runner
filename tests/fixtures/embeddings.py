import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def random_embeddings_01() -> np.array:
    # a random np array with 3 channels, 123 frames
    num_frames = 12
    embeddings = np.random.rand(num_frames, 3, 1280).astype(np.float32)
    hop_size = 5
    offsets = np.repeat(np.arange(0,embeddings.shape[0]*hop_size,hop_size, dtype=np.float32), (3)).reshape(embeddings.shape[0],3,1)
    return np.concatenate((offsets, embeddings), axis=2)


   

# Fixture for generating a random DataFrame
@pytest.fixture
def random_dataframe():
    # Generate a random DataFrame based on your specifications
    # For this test, you need to ensure that this DataFrame
    # is in the correct format and can be converted back into an embeddings array.
    # Example:
    n_segments = 100  # for example
    n_channels = 2   # for example
    n_features = 1281  # for example

    # Generating random data
    data = np.random.rand(n_segments * n_channels, n_features + 2)
    df = pd.DataFrame(data)
    df['channel'] = np.repeat(range(n_channels), n_segments)
    df.columns = ['offset'] + ['channel'] + [f'f{i:04d}' for i in range(n_features)]

    return df