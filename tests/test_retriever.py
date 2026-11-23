"""Tests for rag_chat.retriever (D-013, D-015). No Ollama: embed_query is faked.

Config values are pinned here so tuning floor/δ/w in M4 doesn't break these tests.
"""
import numpy as np
import pytest

from rag_chat import config, retriever
from rag_chat.chunker import Chunk


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    monkeypatch.setattr(config, "SCORE_FLOOR", 0.5)
    monkeypatch.setattr(config, "SCORE_DELTA", 0.2)
    monkeypatch.setattr(config, "MAX_CHUNKS", 8)
    monkeypatch.setattr(config, "QUERY_WEIGHT", 0.8)


def ranked(*scores):
    return [(Chunk("doc.md", ["Doc", f"s{i}"], f"text {i}", 5), s) for i, s in enumerate(scores)]


# --- select ------------------------------------------------------------------

def test_empty_input_gives_empty_output():
    assert retriever.select([]) == []


def test_relative_cutoff_d015_example():
    r = ranked(0.97, 0.82, 0.41, 0.30)                  # cutoff = max(0.5, 0.77) = 0.77
    assert retriever.select(r) == r[:2]


def test_floor_wins_when_top_is_low():
    r = ranked(0.60, 0.52, 0.45)                        # cutoff = max(0.5, 0.40) = 0.5
    assert retriever.select(r) == r[:2]


def test_nothing_relevant_gives_empty_list():
    assert retriever.select(ranked(0.43, 0.42, 0.40)) == []   # D-016 path


def test_at_most_max_chunks():
    assert len(retriever.select(ranked(*[0.9] * 12))) == 8


# --- query_vector ------------------------------------------------------------

@pytest.fixture
def fake_embed(monkeypatch):
    vecs = {"current": np.array([1.0, 0.0]), "previous": np.array([0.0, 1.0])}
    monkeypatch.setattr(retriever.embedder, "embed_query", lambda text: vecs[text])


def test_first_turn_is_just_the_current_question(fake_embed):
    np.testing.assert_array_equal(retriever.query_vector("current", None), [1.0, 0.0])


def test_follow_up_is_weighted_and_renormalized(fake_embed):
    q = retriever.query_vector("current", "previous")
    expected = np.array([0.8, 0.2]) / np.linalg.norm([0.8, 0.2])   # [0.970, 0.243]
    np.testing.assert_allclose(q, expected)
    assert np.isclose(np.linalg.norm(q), 1.0)
