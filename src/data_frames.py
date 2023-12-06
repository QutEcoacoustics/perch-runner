"""
utilities for formatting embeddings into data frames, including multi-channel embeddings
"""

import pandas as pd
import numpy as np
import base64
import struct

def embeddings_to_df(embeddings) -> pd.DataFrame:
    """
    @param embeddings np.array; array of shape (n_segment, n_channels, n_features + 1), where dim 3 is the embeddings for a frame
    plus the frame offset
    @returns pd.DataFrame (n_segments * n_channels) rows and n_features + 2 columns, one col for each feature plus one for offset and one for channel 
    """

    df = pd.DataFrame()

    for channel in range(embeddings.shape[1]):
        array_2d = embeddings[:, channel, :]
        channel_df = pd.DataFrame(array_2d)
        channel_df.insert(0, 'channel', channel)
        # concatenate channel_df to df
        df = pd.concat([df, channel_df], ignore_index=True)

    #rename the columns

    new_columns = ['channel', 'offset'] + embedding_col_names(embeddings.shape[2] - 1)
    df.columns = new_columns

    return df

    
        

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
        encoded_features = serialize_array(features)
        new_row = row[:len(metadata_columns)] + (encoded_features,)

        new_df.loc[len(new_df)] = new_row
        #new_df = new_df.append(pd.Series(new_row, index=new_columns), ignore_index=True)

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



def serialize_array(array) -> str:
    """
    serializes a single clip's embeddings from a 1280 array or list to a string
    using base64 encoding

    """

    #byte_data = b''.join(struct.pack('f', f) for f in array)
    byte_data = np.array(array, dtype=np.float32).tobytes()
    base64_encoded = base64.b64encode(byte_data).decode('utf-8')
    return base64_encoded


def deserialize_array(base64_encoded) -> np.array:
    """
    deserializes a string and produces an array of shape (1280,)

    """

    byte_data = base64.b64decode(base64_encoded)
    #float_data = struct.unpack('f'*int(len(byte_data)/bytes_per_element), byte_data)
    float_data = np.frombuffer(byte_data, dtype=np.float32)
    return np.array(float_data)

    
def embedding_col_names(num_features: int) -> list:
    """
    generates column names for the embedding features columns
    """
    return [f'f{i:04d}' for i in range(num_features)]