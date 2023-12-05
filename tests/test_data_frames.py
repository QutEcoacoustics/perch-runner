
import os
from src import data_frames


def test_embeddings_to_df(random_embeddings_01) -> None:

    result = data_frames.embeddings_to_df(random_embeddings_01)
    expected_shape = (random_embeddings_01.shape[0] * random_embeddings_01.shape[1], random_embeddings_01.shape[2] + 1)
    assert result.shape == expected_shape


def test_serialize_embeddings(random_embeddings_01) -> None:

    embeddings_df = data_frames.embeddings_to_df(random_embeddings_01)
    serialized_embeddings_df = data_frames.serialise_embeddings_df(embeddings_df)
    num_rows, num_columns = serialized_embeddings_df.shape
    # channel, offset, features
    expected_num_columns = 3
    assert num_rows == embeddings_df.shape[0] and num_columns == 3


def test_deserialize_embeddings(random_embeddings_01) -> None:

    embeddings_df = data_frames.embeddings_to_df(random_embeddings_01)
    serialized_embeddings_df = data_frames.serialise_embeddings_df(embeddings_df)
    deserialized_embeddings_df = data_frames.deserialize_embeddings(serialized_embeddings_df)

    assert embeddings_df[1,3] == deserialized_embeddings_df[1,3]


    