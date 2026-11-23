"""Tests for rag_chat.llm (D-010 detection, D-020 streaming). No Ollama: requests.post is faked.

Note: stream() is a generator, so nothing (not even the request) runs until it is iterated.
"""
import json

import pytest
import requests

from rag_chat import config, llm

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


class FakeStream:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = [json.dumps(l).encode() for l in lines]

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ollama_lines(pieces, prompt_eval_count=100, done_reason="stop"):
    lines = [{"message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
    lines.append({"done": True, "done_reason": done_reason, "prompt_eval_count": prompt_eval_count})
    return lines


@pytest.fixture
def fake_ollama(monkeypatch):
    """Call fake_ollama(lines) to set what Ollama streams back; returns the recorded request bodies."""
    sent = []

    def install(lines, status_code=200):
        def fake_post(url, json, stream, timeout):
            sent.append(json)
            return FakeStream(lines, status_code)
        monkeypatch.setattr(llm.requests, "post", fake_post)
        return sent
    return install


def test_yields_pieces_in_order_and_sends_budget_options(fake_ollama, capsys):
    sent = fake_ollama(ollama_lines(["Sync", " runs", " every 5 min."]))
    assert list(llm.stream(MESSAGES, expected=100)) == ["Sync", " runs", " every 5 min."]

    body = sent[0]
    assert body["model"] == config.CHAT_MODEL
    assert body["messages"] == MESSAGES
    assert body["stream"] is True
    assert body["options"] == {"num_ctx": config.TOT_CONTEXT_LENGTH, "num_predict": config.ANS_LENGTH}
    assert capsys.readouterr().err == ""                    # exact match: no warning


def test_fewer_tokens_than_expected_warns_truncated(fake_ollama, capsys):
    fake_ollama(ollama_lines(["ok"], prompt_eval_count=90))
    list(llm.stream(MESSAGES, expected=100))
    assert "TRUNCATED" in capsys.readouterr().err


def test_more_tokens_than_expected_warns_drift(fake_ollama, capsys):
    fake_ollama(ollama_lines(["ok"], prompt_eval_count=110))
    list(llm.stream(MESSAGES, expected=100))
    assert "drift" in capsys.readouterr().err


def test_answer_hitting_num_predict_warns_cut_off(fake_ollama, capsys):
    fake_ollama(ollama_lines(["ok"], done_reason="length"))
    list(llm.stream(MESSAGES, expected=100))
    assert "cut off" in capsys.readouterr().err


def test_error_mid_stream_raises(fake_ollama):
    fake_ollama([{"message": {"content": "par"}, "done": False}, {"error": "model crashed"}])
    with pytest.raises(RuntimeError, match="model crashed"):
        list(llm.stream(MESSAGES, expected=100))


def test_ollama_not_running_gives_clear_error(monkeypatch):
    def refuse(*a, **k):
        raise requests.ConnectionError("connection refused")
    monkeypatch.setattr(llm.requests, "post", refuse)
    with pytest.raises(RuntimeError, match="ollama serve"):
        list(llm.stream(MESSAGES, expected=100))


def test_model_not_pulled_gives_clear_error(fake_ollama):
    fake_ollama([], status_code=404)
    with pytest.raises(RuntimeError, match="ollama pull"):
        list(llm.stream(MESSAGES, expected=100))
