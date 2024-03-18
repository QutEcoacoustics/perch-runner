
import os
import numpy as np
import pytest

from src import data_frames



def test_embeddings_to_df(random_embeddings_01) -> None:

    result = data_frames.embeddings_to_df(random_embeddings_01)
    expected_shape = (random_embeddings_01.shape[0] * random_embeddings_01.shape[1], random_embeddings_01.shape[2] + 1)
    assert result.shape == expected_shape
    assert result.iloc[0,2] == random_embeddings_01[0,0,1]



def test_embeddings_conversion_consistency(random_embeddings_01):
    # Convert embeddings to DataFrame
    df = data_frames.embeddings_to_df(random_embeddings_01)

    # Convert the DataFrame back to embeddings
    converted_embeddings = data_frames.df_to_embeddings(df)

    # Check if the converted embeddings match the original ones
    np.testing.assert_array_almost_equal(random_embeddings_01, converted_embeddings)



def test_serialize_embeddings(random_embeddings_01) -> None:

    embeddings_df = data_frames.embeddings_to_df(random_embeddings_01)
    serialized_embeddings_df = data_frames.serialize_embeddings_df(embeddings_df)
    num_rows, num_columns = serialized_embeddings_df.shape
    # channel, offset, features
    expected_num_columns = 3
    assert num_rows == embeddings_df.shape[0] and num_columns == 3


def test_deserialize_embeddings(random_embeddings_01) -> None:

    embeddings_df = data_frames.embeddings_to_df(random_embeddings_01)
    serialized_embeddings_df = data_frames.serialize_embeddings_df(embeddings_df)
    deserialized_embeddings_df = data_frames.deserialize_embeddings_df(serialized_embeddings_df)

    # checks that all elements of the original match the corresponding element of the deserialized
    assert (embeddings_df == deserialized_embeddings_df).all().all()


def test_serialize_array_wrong_type_1() -> None:

    raw = [1.234, 2.345, 3.456]
    # we always assume 32 bit floats
    with pytest.raises(TypeError, match="Value must be a <class 'numpy.float32'> array, but <class 'list'> was given"):
        data_frames.serialize_array(raw)


def test_serialize_array_wrong_type_2() -> None:

    raw = np.array([1.234, 2.345, 3.456])
    # we always assume 32 bit floats
    with pytest.raises(TypeError, match="Value must be a <class 'numpy.float32'> array, but <class 'numpy.ndarray'> float64 was given"):
        data_frames.serialize_array(raw)


def test_serialize_array() -> None:

    raw = np.array([1.234, 2.345, 3.456], dtype=np.float32)
    #raw = np.array([1.234], dtype=np.float32)
    dtype=np.float32
    #raw = np.array([1], dtype=dtype)
    base64 = 'tvOdP3sUFkAbL11A'
    serialized = data_frames.serialize_array(raw)
    assert serialized == base64
    deserialized = data_frames.deserialize_array(serialized, dtype)
    assert np.array_equal(deserialized, raw)