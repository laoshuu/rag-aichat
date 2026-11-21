"""M0: verify our assumptions about Ollama before building on them.

Run all experiments:      python experiments/ollama_smoke.py
Run selected ones:        python experiments/ollama_smoke.py chat embed overflow

Everything talks to Ollama's REST API with plain `requests` (no client library),
so you can see exactly what goes over the wire. Ollama must be running
(`ollama serve` or `brew services start ollama`).

Ollama API cheat-sheet (all JSON over HTTP, base URL http://localhost:11434):
  POST /api/chat   {"model", "messages": [{"role", "content"}], "stream", "options": {...}}
  POST /api/embed  {"model", "input": str | [str, ...]}          -> {"embeddings": [[...], ...]}
  GET  /api/ps     models currently loaded in memory
  POST /api/show   {"model"} -> metadata (architecture, native context length, template)

Useful "options" for /api/chat:
  num_ctx      context window in tokens (KV cache size). Default is NOT 6144.
  num_predict  max tokens to generate (our answer reserve)
  temperature  0 = deterministic-ish, good for experiments
  seed         fixed seed for reproducibility

All *_duration fields in responses are in NANOSECONDS.
"""

import json
import sys
import time
import uuid

import numpy as np
import requests
from tokenizers import Tokenizer

OLLAMA = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 6144

tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def count(text: str) -> int:
    """Our own token count (what the app will use for budgeting)."""
    return len(tok.encode(text).ids)


def chat(messages, **options) -> dict:
    """One non-streaming /api/chat call. Returns Ollama's full JSON response."""
    body = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, **options},
    }
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def nonce() -> str:
    """A unique string. Putting it at the START of a prompt defeats Ollama's prefix
    cache, so every request is processed from scratch ("cache-cold")."""
    return f"[run {uuid.uuid4().hex[:8]}]"


def filler(target_tokens: int) -> str:
    """Deterministic filler text of roughly `target_tokens` tokens (by our tokenizer)."""
    parts, i = [], 0
    while True:
        parts.append(
            f"Record {i}: the warehouse in district {i} stores {i * 7 % 97} crates of "
            f"apples and {i * 13 % 89} crates of pears, inspected on day {i % 28 + 1}."
        )
        text = "\n".join(parts)
        if count(text) >= target_tokens:
            return text
        i += 1


def stats(resp: dict) -> str:
    """Human-readable summary of the timing/count fields Ollama returns."""
    p_n, p_ns = resp.get("prompt_eval_count", 0), resp.get("prompt_eval_duration", 1)
    e_n, e_ns = resp.get("eval_count", 0), resp.get("eval_duration", 1)
    return (
        f"prompt_eval_count={p_n} ({p_n / (p_ns / 1e9):.0f} tok/s, {p_ns / 1e9:.2f}s) | "
        f"eval_count={e_n} ({e_n / (e_ns / 1e9):.1f} tok/s) | "
        f"load={resp.get('load_duration', 0) / 1e9:.2f}s total={resp.get('total_duration', 0) / 1e9:.2f}s"
    )


def loaded_models() -> list:
    return requests.get(f"{OLLAMA}/api/ps", timeout=10).json()["models"]


def header(title: str):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def exp_show():
    """What does Ollama know about the model? (native context length, template)"""
    header("show: model metadata")
    info = requests.post(f"{OLLAMA}/api/show", json={"model": CHAT_MODEL}, timeout=30).json()
    mi = info.get("model_info", {})
    ctx_keys = {k: v for k, v in mi.items() if "context_length" in k}
    print("native context length:", ctx_keys)
    print("quantization:", info.get("details", {}).get("quantization_level"))
    print("parameters (Modelfile defaults):", info.get("parameters", "(none)"))
    print("chat template (first 400 chars):\n", info.get("template", "")[:400])


def exp_chat():
    """The simplest possible chat call. Look at every field in the response."""
    header("chat: one non-streaming request")
    resp = chat(
        [{"role": "system", "content": "You are concise."},
         {"role": "user", "content": "Name three fruits."}],
        num_ctx=NUM_CTX, num_predict=50,
    )
    content = resp["message"]["content"]
    shown = {k: v for k, v in resp.items() if k != "message"}
    print("answer:", repr(content))
    print("other fields:", json.dumps(shown, indent=2))
    print(stats(resp))


def exp_stream():
    """Streaming: Ollama sends one JSON object per line (NDJSON) as tokens are generated.
    The final line has done=true and carries the stats."""
    header("stream: raw NDJSON lines")
    body = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 5, comma separated."}],
        "stream": True,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 30},
    }
    t0 = time.perf_counter()
    first_token_at = None
    with requests.post(f"{OLLAMA}/api/chat", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for i, line in enumerate(r.iter_lines()):
            msg = json.loads(line)
            if first_token_at is None and msg.get("message", {}).get("content"):
                first_token_at = time.perf_counter() - t0
            if i < 4 or msg.get("done"):
                print(f"line {i}: {line.decode()[:160]}")
            elif i == 4:
                print("   ...")
    print(f"time to first token: {first_token_at:.2f}s")


def exp_embed():
    """Embeddings: batch input, vector size, normalization, and the effect of nomic's prefixes."""
    header("embed: vectors, batching, prefixes")
    docs = [
        "Lumen Notes Guide > Sync > Enable sync\nOpen Settings > Sync and sign in. Notes sync every 5 minutes.",
        "Lumen Notes Guide > Install\nDownload Lumen from lumen.example and drag it to Applications.",
        "team_notes.txt\nLumen Pro will cost $4 per month. Launch date moved to May 12.",
    ]
    query = "How do I turn on sync?"

    def embed(texts):
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=120)
        r.raise_for_status()
        return r.json()

    resp = embed(["search_document: " + d for d in docs])
    vecs = np.array(resp["embeddings"])
    print("shape:", vecs.shape, "| L2 norms:", np.round(np.linalg.norm(vecs, axis=1), 4))
    print("response fields:", [k for k in resp if k != "embeddings"])

    for label, q_prefix, d_prefix in [("WITH prefixes", "search_query: ", "search_document: "),
                                      ("WITHOUT prefixes", "", "")]:
        d = np.array(embed([d_prefix + x for x in docs])["embeddings"])
        q = np.array(embed([q_prefix + query])["embeddings"])[0]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        q /= np.linalg.norm(q)
        scores = d @ q
        print(f"{label:17s} cosine scores:", np.round(scores, 3), "(sync, install, pricing)")

    batch = [f"search_document: chunk number {i}. " + docs[i % 3] * 6 for i in range(64)]
    avg_tokens = np.mean([count(b) for b in batch])
    t0 = time.perf_counter()
    embed(batch)
    dt = time.perf_counter() - t0
    print(f"batch of 64 chunks (~{avg_tokens:.0f} tok each): {dt:.2f}s -> {64 / dt:.0f} chunks/s")


def exp_template_overhead():
    """How many extra tokens does the chat template add per message?
    Compare our count of the message *contents* with Ollama's prompt_eval_count."""
    header("template overhead: our count vs prompt_eval_count")
    base = [{"role": "system", "content": f"{nonce()} You answer from context only."}]
    turns = [
        {"role": "user", "content": "How do I turn on sync?"},
        {"role": "assistant", "content": "Open Settings > Sync and sign in."},
        {"role": "user", "content": "Does it have any limits?"},
        {"role": "assistant", "content": "Free accounts can sync up to 3 devices."},
        {"role": "user", "content": "When does it launch?"},
    ]
    rows = []
    for n in (1, 3, 5):
        msgs = [{**base[0], "content": f"{nonce()} You answer from context only."}] + turns[:n]
        ours = sum(count(m["content"]) for m in msgs)
        resp = chat(msgs, num_ctx=NUM_CTX, num_predict=1)
        theirs = resp["prompt_eval_count"]
        rows.append((len(msgs), ours, theirs))
        print(f"{len(msgs)} messages: our content count={ours:4d}  prompt_eval_count={theirs:4d}  "
              f"overhead={theirs - ours:3d}  ({(theirs - ours) / len(msgs):.1f}/message)")
    # Without a system message, Qwen's template injects its own default system prompt.
    msgs = [{"role": "user", "content": f"{nonce()} How do I turn on sync?"}]
    resp = chat(msgs, num_ctx=NUM_CTX, num_predict=1)
    print(f"no system message: our count={count(msgs[0]['content'])} "
          f"prompt_eval_count={resp['prompt_eval_count']}  <- difference includes Qwen's default system prompt")


def exp_default_ctx():
    """What context window does Ollama use if we DON'T set num_ctx?"""
    header("default context: ~5,000-token prompt WITHOUT num_ctx")
    secret = "The secret code word is BLUEBERRY."
    text = f"{nonce()} {secret}\n" + filler(5000)
    question = "\n\nWhat is the secret code word? Answer with just the word."
    msgs = [{"role": "user", "content": text + question}]
    ours = count(msgs[0]["content"])
    resp = chat(msgs, num_predict=10)  # note: no num_ctx
    print(f"our count={ours}  prompt_eval_count={resp['prompt_eval_count']}")
    print("answer:", repr(resp["message"]["content"]))
    print("loaded:", [(m["name"], m.get("context_length")) for m in loaded_models()])
    print(stats(resp))


def exp_num_ctx():
    """Same ~5,000-token prompt WITH num_ctx=6144: everything should be processed.
    Also our main cache-cold prefill speed measurement."""
    header(f"num_ctx={NUM_CTX}: ~5,000-token prompt")
    secret = "The secret code word is BLUEBERRY."
    text = f"{nonce()} {secret}\n" + filler(5000)
    question = "\n\nWhat is the secret code word? Answer with just the word."
    msgs = [{"role": "user", "content": text + question}]
    ours = count(msgs[0]["content"])
    resp = chat(msgs, num_ctx=NUM_CTX, num_predict=10)
    print(f"our count={ours}  prompt_eval_count={resp['prompt_eval_count']}")
    print("answer:", repr(resp["message"]["content"]))
    print("loaded:", [(m["name"], m.get("context_length")) for m in loaded_models()])
    print(stats(resp))


def exp_overflow():
    """Send MORE than num_ctx tokens. Does Ollama error, or truncate silently? What gets cut?
    Secret at the START, question at the END. Watch the `ollama serve` terminal for warnings."""
    header(f"overflow: ~7,000-token prompt with num_ctx={NUM_CTX}")
    secret = "The secret code word is BLUEBERRY."
    text = f"{nonce()} {secret}\n" + filler(7000)
    question = "\n\nWhat is the secret code word? Answer with just the word."
    msgs = [
        {"role": "system", "content": "Always answer in ALL CAPITAL LETTERS."},
        {"role": "user", "content": text + question},
    ]
    ours = sum(count(m["content"]) for m in msgs)
    try:
        resp = chat(msgs, num_ctx=NUM_CTX, num_predict=20)
    except requests.HTTPError as e:
        print("Ollama returned an ERROR (loud failure):", e, e.response.text[:300])
        return
    print(f"our count={ours}  prompt_eval_count={resp['prompt_eval_count']}  (num_ctx={NUM_CTX})")
    print("answer:", repr(resp["message"]["content"]))
    print("  -> knows the secret?  ", "BLUEBERRY" in resp["message"]["content"].upper())
    print("  -> followed system prompt (all caps)?",
          resp["message"]["content"].strip() == resp["message"]["content"].strip().upper())
    print(stats(resp))


def exp_cache():
    """Send the SAME ~3,000-token prompt twice. Does prompt_eval_count report the full
    prompt the second time, or only the newly processed tokens?"""
    header("cache: identical prompt twice, then same prefix + different question")
    prefix = f"{nonce()}\n" + filler(3000)
    q1 = [{"role": "user", "content": prefix + "\n\nHow many records are there?"}]
    q2 = [{"role": "user", "content": prefix + "\n\nWhich district has the most apples?"}]
    print(f"our count: q1={count(q1[0]['content'])}  q2={count(q2[0]['content'])}")
    for label, msgs in [("1st (cold)", q1), ("2nd (identical)", q1), ("3rd (same prefix)", q2)]:
        resp = chat(msgs, num_ctx=NUM_CTX, num_predict=5)
        print(f"{label:18s} {stats(resp)}")


def exp_decode_speed():
    """Generation speed for a realistic ~300-token answer."""
    header("decode speed: ~300-token answer")
    resp = chat(
        [{"role": "user", "content": f"{nonce()} Write a 250-word explanation of how a library catalogue works."}],
        num_ctx=NUM_CTX, num_predict=400,
    )
    print(stats(resp))


def exp_reload():
    """Does changing num_ctx between requests force a model reload? (look at load_duration)"""
    header("reload: alternating num_ctx")
    for n in (NUM_CTX, NUM_CTX, 4096, NUM_CTX):
        resp = chat([{"role": "user", "content": f"{nonce()} Say OK."}], num_ctx=n, num_predict=3)
        print(f"num_ctx={n:5d}  load_duration={resp['load_duration'] / 1e9:.2f}s")


def exp_truncation():
    """Follow-up to `overflow`: what EXACTLY does Ollama keep when a prompt is too long?
    Hypothesis A: keeps half the PROMPT.  Hypothesis B: keeps half the WINDOW (~3,074 at 6,144).
    Four code words sit at ~0%, 35%, 65% and 95% of the prompt, so the model's answer
    shows which regions survived."""
    header(f"truncation: vary prompt size at num_ctx={NUM_CTX}; which parts survive?")
    words = ["APPLE", "RIVER", "CANDLE", "TIGER"]
    question = "\n\nList every code word mentioned above, in order. Reply with just the words."
    for target in (6000, 6300, 7000, 9000, 12000):
        lines = filler(target).split("\n")
        for k, frac in reversed(list(enumerate((0.0, 0.35, 0.65, 0.95)))):
            lines.insert(int(frac * len(lines)), f"Code word number {k + 1} is {words[k]}.")
        text = f"{nonce()}\n" + "\n".join(lines) + question
        ours = count(text)
        resp = chat([{"role": "user", "content": text}], num_ctx=NUM_CTX, num_predict=30)
        ans = resp["message"]["content"].upper()
        seen = [w for w in words if w in ans]
        print(f"sent={ours:6d}  kept={resp['prompt_eval_count']:5d}  "
              f"prompt/2={ours // 2:5d}  window/2={NUM_CTX // 2}  survived={seen}")

    # Edge case: the prompt FITS, but prompt + answer does not.
    text = f"{nonce()}\n" + filler(6100) + "\n\nWrite a long story about these warehouses."
    resp = chat([{"role": "user", "content": text}], num_ctx=NUM_CTX, num_predict=200)
    print(f"\nprompt fits but no room to answer: sent={count(text)} kept={resp['prompt_eval_count']} "
          f"generated={resp['eval_count']} done_reason={resp.get('done_reason')}")


def exp_cache_prefix():
    """How does the prefix cache match? Change ONE line at different depths of a cached
    prompt and see how much is reused. The final fully-new prompt is the control: if it
    is slow again, the speed-ups came from the cache, not from 'warm-up'."""
    header("cache prefix: change one line at 75% / 25% depth, then a fully new prompt")
    head = nonce()
    lines = filler(3000).split("\n")
    q = "\n\nHow many records are there?"

    def build(change_at=None):
        ls = list(lines)
        if change_at is not None:
            i = int(change_at * len(ls))
            ls[i] = ls[i].replace("apples", "plums")   # a one-word change
        return f"{head}\n" + "\n".join(ls) + q

    runs = [("1 cold", build()), ("2 identical", build()),
            ("3 change @75%", build(0.75)), ("4 change @25%", build(0.25))]
    for label, text in runs:
        r = chat([{"role": "user", "content": text}], num_ctx=NUM_CTX, num_predict=3)
        print(f"{label:16s} total={r['prompt_eval_count']:5d} cached={r.get('prompt_eval_cached_count'):5d} "
              f"prefill={r['prompt_eval_duration'] / 1e9:6.2f}s")
    head = nonce()   # control: brand-new prefix, same length
    r = chat([{"role": "user", "content": build()}], num_ctx=NUM_CTX, num_predict=3)
    print(f"{'5 new (control)':16s} total={r['prompt_eval_count']:5d} cached={r.get('prompt_eval_cached_count'):5d} "
          f"prefill={r['prompt_eval_duration'] / 1e9:6.2f}s")


EXPERIMENTS = {
    "show": exp_show,
    "chat": exp_chat,
    "stream": exp_stream,
    "embed": exp_embed,
    "overhead": exp_template_overhead,
    "default_ctx": exp_default_ctx,
    "num_ctx": exp_num_ctx,
    "overflow": exp_overflow,
    "cache": exp_cache,
    "decode": exp_decode_speed,
    "reload": exp_reload,
    "truncation": exp_truncation,
    "cache_prefix": exp_cache_prefix,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        EXPERIMENTS[name]()
