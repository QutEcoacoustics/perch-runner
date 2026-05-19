import numpy as np
import pandas as pd
import pytest

from src import data_frames


# ---------------------------------------------------------------------------
# serialize_array / deserialize_array
# ---------------------------------------------------------------------------

class TestSerializeArray:

    def test_round_trip_float32(self):
        raw = np.array([1.234, 2.345, 3.456], dtype=np.float32)
        serialized = data_frames.serialize_array(raw)
        deserialized = data_frames.deserialize_array(serialized)
        np.testing.assert_array_equal(deserialized, raw)

    def test_known_value(self):
        """Verify a specific known encoding to ensure stability."""
        raw = np.array([1.234, 2.345, 3.456], dtype=np.float32)
        serialized = data_frames.serialize_array(raw)
        assert serialized == "tvOdP3sUFkAbL11A"

    def test_empty_array(self):
        raw = np.array([], dtype=np.float32)
        serialized = data_frames.serialize_array(raw)
        deserialized = data_frames.deserialize_array(serialized)
        assert len(deserialized) == 0

    def test_large_array(self):
        raw = np.random.rand(1536).astype(np.float32)
        serialized = data_frames.serialize_array(raw)
        deserialized = data_frames.deserialize_array(serialized)
        np.testing.assert_array_equal(deserialized, raw)

    def test_rejects_list(self):
        with pytest.raises(TypeError, match="Value must be"):
            data_frames.serialize_array([1.234, 2.345, 3.456])

    def test_rejects_float64(self):
        raw = np.array([1.234, 2.345, 3.456])  # default float64
        with pytest.raises(TypeError, match="Value must be"):
            data_frames.serialize_array(raw)

    def test_explicit_dtype_float64(self):
        raw = np.array([1.234, 2.345], dtype=np.float64)
        serialized = data_frames.serialize_array(raw, dtype=np.float64)
        deserialized = data_frames.deserialize_array(serialized, dtype=np.float64)
        np.testing.assert_array_equal(deserialized, raw)


# ---------------------------------------------------------------------------
# embedding_col_names
# ---------------------------------------------------------------------------

class TestEmbeddingColNames:

    def test_basic(self):
        names = data_frames.embedding_col_names(3)
        assert names == ["f0000", "f0001", "f0002"]

    def test_zero(self):
        assert data_frames.embedding_col_names(0) == []

    def test_large(self):
        names = data_frames.embedding_col_names(1536)
        assert len(names) == 1536
        assert names[0] == "f0000"
        assert names[-1] == "f1535"

    def test_consistent_width(self):
        """All names should be the same length for clean column alignment."""
        names = data_frames.embedding_col_names(100)
        lengths = {len(n) for n in names}
        assert len(lengths) == 1


# ---------------------------------------------------------------------------
# serialize_embeddings_df / deserialize_embeddings_df
# ---------------------------------------------------------------------------

class TestSerializeEmbeddingsDf:

    @pytest.fixture
    def columns_df(self):
        """A DataFrame in columns format (f0000, f0001, ...) with metadata."""
        n_rows = 5
        n_features = 10
        col_names = data_frames.embedding_col_names(n_features)
        data = np.random.rand(n_rows, n_features).astype(np.float32)
        df = pd.DataFrame(data, columns=col_names)
        df.insert(0, "channel", 0)
        df.insert(1, "offset", np.arange(n_rows) * 5.0)
        return df

    def test_serialize_shape(self, columns_df):
        result = data_frames.serialize_embeddings_df(columns_df)
        assert result.shape == (len(columns_df), 3)
        assert list(result.columns) == ["channel", "offset", "embeddings"]

    def test_round_trip(self, columns_df):
        serialized = data_frames.serialize_embeddings_df(columns_df)
        deserialized = data_frames.deserialize_embeddings_df(serialized)
        # Compare values (column names and dtypes may differ slightly)
        np.testing.assert_array_almost_equal(
            deserialized.iloc[:, 2:].values.astype(np.float32),
            columns_df.iloc[:, 2:].values,
        )

    def test_preserves_metadata(self, columns_df):
        serialized = data_frames.serialize_embeddings_df(columns_df)
        assert list(serialized["channel"]) == list(columns_df["channel"])
        assert list(serialized["offset"]) == list(columns_df["offset"])

    def test_invalid_metadata_columns(self, columns_df):
        with pytest.raises(ValueError, match="metadata columns"):
            data_frames.serialize_embeddings_df(columns_df, metadata_columns=("nonexistent",))

    def test_deserialize_invalid_col(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="embeddings column"):
            data_frames.deserialize_embeddings_df(df)
