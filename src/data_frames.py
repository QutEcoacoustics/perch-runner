"""
utilities for formatting embeddings into data frames
"""

import pandas as pd
import numpy as np
import base64


def serialize_embeddings_df(df, metadata_columns = ('channel', 'offset')) -> pd.DataFrame:
    """
    Converts a dataframe of embeddings with 1 embedding feature per column
    to a dataframe that has 1 column for all embeddings as a base64 encoded string
    """
    
    if not all(col in df.columns for col in metadata_columns):
        raise ValueError("supplied metadata columns are not in the given dataframe")

    new_columns = list(metadata_columns) + ["embeddings"]
    new_df = pd.DataFrame(columns=new_columns)

    feature_columns = [col for col in df.columns if col not in metadata_columns]


    for row in df.itertuples(index=False):

        features = row[len(metadata_columns):]
        encoded_features = serialize_array(np.array(features, dtype=np.float32))
        new_row = row[:len(metadata_columns)] + (encoded_features,)

        new_df.loc[len(new_df)] = new_row

    return new_df


def deserialize_embeddings_df(df, embedding_col = 'embeddings') -> pd.DataFrame:
    """
    Converts a dataframe of embeddings with 1 columns for all embedding features as base64 encoded
    array of floats to 1 column per embedding feature as float
    """
    
    if not embedding_col in df.columns:
        raise ValueError("supplied embeddings column is not in the given dataframe")
    
    # deserialize the 1st row to get the number of feature columns
    embeddings_0 = deserialize_array(df["embeddings"][0])

    metadata_columns = list(df.columns)
    metadata_columns.remove(embedding_col)

    new_columns = metadata_columns + embedding_col_names(len(embeddings_0))
    new_df = pd.DataFrame(columns=new_columns)


    for row in df.itertuples(index=False):

        metadata = [getattr(row, key) for key in metadata_columns]
        serialized_embeddings = getattr(row, embedding_col)
        raw_embeddings = list(deserialize_array(serialized_embeddings))
        new_df.loc[len(new_df)] = metadata + raw_embeddings

    return new_df


def serialize_array(array: np.ndarray, dtype=np.float32) -> str:
    """
    serializes a single clip's embeddings from a 1280 array or list to a string
    using base64 encoding
    """

    if not isinstance(array, np.ndarray) or not array.dtype == dtype:
        supplied_type = f'{type(array)} {array.dtype}' if isinstance(array, np.ndarray) else type(array)
        raise TypeError(f"Value must be a {dtype} array, but {supplied_type} was given")

    bytes = array.tobytes()
    base64_encoded = base64.b64encode(bytes).decode('ascii')
    return base64_encoded

def deserialize_array(base64_encoded, dtype=np.float32) -> np.ndarray:
    """
    deserializes a base64-encoded string back into a numpy array
    """

    byte_data = base64.b64decode(base64_encoded)
    float_data = np.frombuffer(byte_data, dtype=dtype, count=-1, offset=0)
    return float_data

    
def embedding_col_names(num_features: int) -> list:
    """
    generates column names for the embedding features columns
    """
    return [f'f{i:04d}' for i in range(num_features)]