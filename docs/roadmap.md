# Roadmap

**Review:** Mon Nov 24, 1:00 PM. **Repo must be shared by Sun Nov 23, 1:00 PM**, which is our real deadline.
**Target code freeze:** Sun Nov 23, 10:00 AM (a 3-hour buffer before sharing).

We build in the order data flows through the system. Each milestone ends with something we can **run and see**, then we commit, and **push after every milestone**, so a working version is always on GitHub.

```
M0 verify Ollama ─► M1 text → chunks ─► M2 chunks → index/ ─► M3 index → chat ─► M4 demo-ready
   (experiments)      (no Ollama)          (ingest works)        (chat works)       (corpus, eval, README)
```

## Schedule

| When | Milestone |
|---|---|
| Thu Nov 20 | **M0** (Claude runs the experiments; Ryan learns the Ollama API from the code) · start **M1** |
| Fri Nov 21 | Finish **M1** · **M2** |
| Sat Nov 22 | **M3** (end-to-end chat) · start **M4** |
| Sun Nov 23 AM | Finish **M4** essentials · freeze 10:00 · share by 13:00 |
| Sun Nov 23 PM – Mon AM | Review prep: demo rehearsal, re-read decisions.md, practise reviewer questions |

**Cut-line if time runs short:** M0–M3 plus a minimal corpus and README is a complete submission. A small eval (even 8–10 questions) is what makes the tuning numbers defensible, so keep at least that much of M4.

---

## M0: check our assumptions about Ollama *(experiments only)*

| File | Checks |
|---|---|
| `experiments/ollama_smoke.py` | Basic chat and embed calls and their response fields · default context size · `num_ctx=6144` respected · what overflow does · `prompt_eval_count` on cached repeats · per-message template overhead vs our tokenizer · prefill and decode speed · embedding speed |

**Done when:** every `[verify]` tag in decisions.md is replaced by a measured result, recorded in `experiments/README.md`.

## M1: text → chunks *(pure Python, no Ollama)*

The hardest logic in the project, so it comes first.

| File | Job | Author |
|---|---|---|
| `requirements.txt`, `rag_chat/__init__.py` | Scaffolding | Claude |
| `rag_chat/config.py` | All constants (`NUM_CTX`, budget caps, model names, chunk size) | Ryan |
| `rag_chat/tokens.py` | `count(text)` with the Qwen tokenizer + per-message overhead | Ryan |
| `rag_chat/loaders.py` | Find and read `.md`/`.txt`; handle bad encodings and empty files | Ryan |
| `rag_chat/chunker.py` | Headings → paragraphs → sentences; ~400-token packing; overlap; heading path | Ryan (**the main piece**) |
| `tests/test_chunker.py`, `tests/test_loaders.py` | Edge cases | Ryan (Claude suggests cases) |

**Done when** the Lumen example prints its 4 chunks with heading paths and token counts, and tests pass for: `#` inside a code fence, a 2,000-token section, an empty file, a `.txt` with no headings.

## M2: chunks → `index/` *(`ingest` works)*

| File | Job |
|---|---|
| `rag_chat/embedder.py` | Batched `/api/embed`; `search_document:` / `search_query:` prefixes; L2-normalization |
| `rag_chat/store.py` | Save and load `vectors.npy` + `chunks.json` + `manifest.json`; `search(q)`; model-mismatch check |
| `rag_chat/cli.py` (`ingest`) | `python -m rag_chat ingest data/` |
| `tests/test_store.py` | Save → load round trip; known vector ranks first |

**Done when** `ingest` writes `index/` and a query prints sensible ranked chunks, with **no LLM involved yet**. Check retrieval quality here.

## M3: index → chat *(end-to-end)*

| File | Job |
|---|---|
| `rag_chat/retriever.py` | Weighted query vector (D-013); relevance floor + relative cutoff + budget (D-015) |
| `rag_chat/prompt.py` | System + history window + context + question; **≤ 5,120 prompt tokens** |
| `rag_chat/llm.py` | Streaming `/api/chat`; `num_ctx`; `prompt_eval_count` check |
| `rag_chat/cli.py` (`chat`) | REPL, streaming output, sources, `/quit`, `/reset`, model warm-up |
| `tests/test_prompt.py` | **Key test:** whatever chunks or history are fed in, the prompt never exceeds the budget |

**Done when** the Lumen walkthrough questions, including the follow-up, work live.

## M4: demo-ready

| File | Job |
|---|---|
| `data/` | A permissively licensed public Markdown doc set + 3 fictional `.txt` notes |
| `eval/questions.jsonl`, `eval/run_eval.py` | ~15 questions (single, follow-up, topic switch, unanswerable) → hit-rate@k; tune `w`, floor, δ, chunk size |
| `README.md` | Clone → install Ollama → pull models → venv → ingest → chat; architecture summary |
| `docs/decisions.md` | Replace guesses with measured numbers |
