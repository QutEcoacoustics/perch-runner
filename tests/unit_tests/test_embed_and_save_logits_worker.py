import numpy as np

from src.embed_and_save_logits_worker import _select_top_indices_above_threshold


def test_select_top_indices_applies_threshold_and_top_n():
    logits = np.array([0.50, 0.95, 0.10, 0.80, 0.70], dtype=np.float32)

    got = _select_top_indices_above_threshold(logits, threshold=0.6, top_n=2)

    # Eligible indices are [1, 3, 4], top-2 by score are [1, 3].
    assert got.tolist() == [1, 3]


def test_select_top_indices_returns_all_when_fewer_than_top_n():
    logits = np.array([0.9, 0.2, 0.7], dtype=np.float32)

    got = _select_top_indices_above_threshold(logits, threshold=0.5, top_n=10)

    assert got.tolist() == [0, 2]


def test_select_top_indices_returns_empty_when_no_scores_meet_threshold():
    logits = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    got = _select_top_indices_above_threshold(logits, threshold=0.5, top_n=10)

    assert got.size == 0
