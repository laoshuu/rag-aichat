import json
import sys
import requests
from rag_chat import config
from collections.abc import Iterator

URL = f"{config.OLLAMA_BASE_URL}/api/chat"


def stream(messages: list[dict], expected: int) -> Iterator[str]:
    body = {
        "model":    config.CHAT_MODEL,
        "messages": messages,
        "stream":   True,
        "options":  {"num_ctx": config.TOT_CONTEXT_LENGTH,     # 6,144 on every request
                     "num_predict": config.ANS_LENGTH},        # 1,024: the answer slot
    }
    try:
        resp = requests.post(URL, json=body, stream=True, timeout=120)  # per-read timeout; prefill ≤ ~19 s
    except requests.ConnectionError:
        raise RuntimeError("Ollama not reachable. Run `ollama serve`.")
    if resp.status_code == 404:
        raise RuntimeError(f"Model {config.CHAT_MODEL} not found. Run `ollama pull {config.CHAT_MODEL}`.")
    resp.raise_for_status()

    with resp:
        for line in resp.iter_lines():            # NDJSON: one JSON object per line
            if not line: 
                continue
            msg = json.loads(line)
            if "error" in msg:                    # Ollama can fail mid-stream
                raise RuntimeError(msg["error"])
            if msg["done"]:
                _check(msg, expected)             # last line carries the stats
                return
            yield msg["message"]["content"]       # a few characters at a time

def _check(final: dict, expected: int):
    # D-010 detection: our exact count vs what Ollama actually processed
    actual = final["prompt_eval_count"]
    if actual < expected:
        print(f"\nwarning: Ollama processed {actual} prompt tokens, expected {expected}: prompt was TRUNCATED", file=sys.stderr)
    elif actual != expected:
        print(f"\nwarning: token count drift: ours {expected}, Ollama {actual}; tokenizer/template mismatch?", file=sys.stderr)
    if final.get("done_reason") == "length":      # hit num_predict
        print(f"\nwarning: answer cut off at 1,024 tokens", file=sys.stderr)