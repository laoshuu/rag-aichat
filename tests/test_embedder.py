"""Tests for rag_chat.embedder (D-007). No Ollama: requests.post is faked.

What we prove is the two silent-failure modes (wrong/missing prefix, no
normalization), plus batching and the two first-run error messages.
"""
import numpy as np
import pytest
import requests

from rag_chat import config, embedder


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_ollama(monkeypatch):
    """Replace requests.post; record every input batch; return [3, 4] per text.

    [3, 4] has norm 5, so a norm-1 result proves OUR code normalized it,
    not Ollama (real nomic vectors already come back normalized).
    """
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json["input"])
        return FakeResponse(payload={"embeddings": [[3.0, 4.0]] * len(json["input"])})

    monkeypatch.setattr(embedder.requests, "post", fake_post)
    return calls


def test_documents_get_doc_prefix_and_are_normalized(fake_ollama):
    vecs = embedder.embed_documents(["Guide > Sync\nLumen syncs.", "Team notes"])
    assert fake_ollama == [[
        "search_document: Guide > Sync\nLumen syncs.",
        "search_document: Team notes",
    ]]
    assert vecs.shape == (2, 2)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(vecs, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)


def test_query_gets_query_prefix_and_is_1d_normalized(fake_ollama):
    q = embedder.embed_query("how often does sync run?")
    assert fake_ollama == [["search_query: how often does sync run?"]]
    assert q.shape == (2,)                               # 1-D, so M @ q gives N scores
    np.testing.assert_allclose(q, [0.6, 0.8], rtol=1e-6)


def test_documents_are_sent_in_batches_in_order(fake_ollama, monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 2)
    vecs = embedder.embed_documents(["a", "b", "c", "d", "e"])
    assert [len(batch) for batch in fake_ollama] == [2, 2, 1]   # short last batch
    assert [t for batch in fake_ollama for t in batch] == [
        f"search_document: {t}" for t in "abcde"                # order kept: row i = chunk i
    ]
    assert vecs.shape == (5, 2)


def test_ollama_not_running_gives_clear_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(embedder.requests, "post", refuse)
    with pytest.raises(RuntimeError, match="ollama serve"):
        embedder.embed_query("hi")


def test_model_not_pulled_gives_clear_error(monkeypatch):
    # Real Ollama answer for an unknown model (verified with curl): 404.
    monkeypatch.setattr(
        embedder.requests, "post",
        lambda *a, **k: FakeResponse(404, {"error": "model 'x' not found"}),
    )
    with pytest.raises(RuntimeError, match="ollama pull"):
        embedder.embed_query("hi")
