"""Tests for rag_chat.tokens (D-010).

Unit tests pin the tokenizer's behaviour and the overhead formula.
The last test checks the formula against Ollama itself (ground truth) and is
skipped when Ollama or the chat model isn't available.
"""
import pytest
import requests

from rag_chat import config
from rag_chat.tokens import count, count_messages

OVERHEAD_PER_MSG = 5  # measured in M0 (experiments/README.md, finding 4)
TRAILER = 3           # trailing <|im_start|>assistant\n


# --- count() -----------------------------------------------------------------

def test_empty_string_is_zero_tokens():
    assert count("") == 0


def test_digits_are_split_one_token_each():
    # M0 / D-010: numbers tokenize digit by digit -> "128K" is 1,2,8,K.
    assert count("128K") == 4


def test_plain_words():
    assert count("hello world") == 2


def test_chatml_marker_is_one_special_token():
    # D-010: <|im_start|> is a single special token (id 151644), not ~6 pieces.
    assert count("<|im_start|>") == 1


def test_no_special_tokens_added():
    # Content-only count: no BOS/EOS padding, so counts are additive for
    # simple word-boundary joins.
    assert count("hello") + count(" world") == count("hello world")


# --- count_messages() --------------------------------------------------------

def test_no_messages_is_just_the_trailer():
    assert count_messages([]) == TRAILER


def test_one_empty_message_costs_the_wrapper():
    assert count_messages([{"role": "user", "content": ""}]) == OVERHEAD_PER_MSG + TRAILER


def test_formula_on_a_real_conversation():
    msgs = [
        {"role": "system", "content": "You answer from context only."},
        {"role": "user", "content": "How do I turn on sync?"},
        {"role": "assistant", "content": "Open Settings > Sync and sign in."},
    ]
    content = sum(count(m["content"]) for m in msgs)
    assert count_messages(msgs) == content + OVERHEAD_PER_MSG * len(msgs) + TRAILER


# --- Ground truth: our count == Ollama's prompt_eval_count -------------------

def _ollama_chat(msgs):
    """Return Ollama's response, or None if Ollama / the model isn't available."""
    try:
        r = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.CHAT_MODEL,
                "messages": msgs,
                "stream": False,
                "options": {"num_ctx": config.TOT_CONTEXT_LENGTH, "num_predict": 1},
            },
            timeout=120,
        )
    except requests.ConnectionError:
        return None
    if r.status_code == 404:  # model not pulled
        return None
    r.raise_for_status()
    return r.json()


def test_count_matches_ollama_prompt_eval_count():
    msgs = [
        {"role": "system", "content": "You answer from context only."},
        {"role": "user", "content": "How do I turn on sync?"},
        {"role": "assistant", "content": "Open Settings > Sync and sign in."},
        {"role": "user", "content": "Does it work on 128K files? 本地运行吗？"},
    ]
    resp = _ollama_chat(msgs)
    if resp is None:
        pytest.skip(f"Ollama or {config.CHAT_MODEL} not available")
    # prompt_eval_count reports the full prompt even when cached (D-010, M0).
    assert count_messages(msgs) == resp["prompt_eval_count"]
