"""Tests for rag_chat.prompt (D-009, D-011, D-015, D-016). No Ollama: real Qwen tokenizer.

The key property: whatever history, chunks or question go in, the prompt
never exceeds PROMPT_LIMIT (5,120) and `expected` is the exact count.
"""
import random

import pytest

from rag_chat import config, prompt, tokens
from rag_chat.chunker import Chunk


def make_chunk(name: str, n_words: int) -> Chunk:
    text = " ".join(f"{name}{i}" for i in range(n_words))
    return Chunk(f"{name}.md", ["Guide", name], text, tokens.count(f"Guide > {name}\n{text}"))


def turn(n_words: int, tag: str = "t") -> tuple[str, str]:
    return (f"question {tag}?", " ".join(f"{tag}{i}" for i in range(n_words)))


# --- the key test: the budget always holds ----------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_prompt_never_exceeds_limit(seed):
    rng = random.Random(seed)
    history = [turn(rng.randint(10, 600), f"h{i}") for i in range(rng.randint(0, 15))]
    ranked  = [(make_chunk(f"c{i}", rng.randint(50, 500)), 0.9 - i * 0.01) for i in range(20)]
    current = " ".join(["why"] * rng.randint(1, 1500))     # up to a pasted-text question

    messages, included, expected = prompt.build(history, ranked, current)

    assert expected <= prompt.PROMPT_LIMIT
    assert expected == tokens.count_messages(messages)      # what llm.py compares with Ollama
    assert tokens.count(messages[-1]["content"]) <= config.RETRIEVAL_LENGTH
    assert tokens.count_messages(messages[1:-1]) - tokens.TRAILER <= config.HISTORY_LENGTH


def test_system_prompt_fits_its_cap():
    assert tokens.count(prompt.SYSTEM_PROMPT) <= config.SYSTEM_PROMPT_LENGTH


# --- shape of the messages ----------------------------------------------------

def test_order_is_system_history_then_context_and_question():
    c = make_chunk("sync", 20)
    history = [("How do I turn on sync?", "Open Settings > Sync.")]
    messages, _, _ = prompt.build(history, [(c, 0.8)], "What about its limits?")

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == prompt.SYSTEM_PROMPT
    assert messages[1]["content"] == "How do I turn on sync?"      # raw question, no old context
    last = messages[-1]["content"]
    assert c.full_text in last and last.endswith("Question: What about its limits?")


def test_no_chunks_tells_model_there_are_no_documents():
    messages, included, _ = prompt.build([], [], "When is the launch?")
    assert included == []
    assert "(no relevant documents found)" in messages[-1]["content"]


# --- chunk fitting (D-015) ----------------------------------------------------

def test_chunk_that_does_not_fit_is_skipped_and_next_one_tried(monkeypatch):
    a, b, c = make_chunk("a", 60), make_chunk("b", 300), make_chunk("c", 20)
    q = "What about its limits?"
    # Budget fits exactly a + c, so b (too big) must be skipped and c still taken.
    monkeypatch.setattr(config, "RETRIEVAL_LENGTH", tokens.count(prompt.query_message([a, c], q)))

    _, included, _ = prompt.build([], [(a, 0.9), (b, 0.8), (c, 0.7)], q)
    assert included == [(a, 0.9), (c, 0.7)]                 # scores kept for the source list


def test_question_alone_too_long_raises(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_LENGTH", 10)
    with pytest.raises(ValueError, match="too long"):
        prompt.build([], [], "a question that is clearly longer than ten tokens in total")


# --- history window (D-009) ---------------------------------------------------

def test_window_keeps_newest_turns_in_order():
    history = [turn(5, "old"), turn(5, "mid"), turn(5, "new")]
    msgs = prompt.window(history)
    assert [m["content"] for m in msgs[::2]] == ["question old?", "question mid?", "question new?"]


def test_window_stops_at_first_turn_that_does_not_fit():
    # small, HUGE, small: the oldest small turn must NOT be kept (no gaps)
    history = [turn(5, "old"), turn(2000, "huge"), turn(5, "new")]
    msgs = prompt.window(history)
    assert [m["content"] for m in msgs[::2]] == ["question new?"]


def test_window_drops_a_single_turn_bigger_than_the_cap():
    assert prompt.window([turn(2000, "huge")]) == []
