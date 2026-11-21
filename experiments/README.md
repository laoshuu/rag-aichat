# Experiments

Small scripts that check assumptions with real measurements before we build on them. Each result here is cited in [`docs/decisions.md`](../docs/decisions.md).

| Script | Question |
|---|---|
| `tokenizer_ratios.py` | How many characters per token does Qwen2.5 use for different content types? |
| `ollama_smoke.py` | How does Ollama actually behave (context size, overflow, caching, speed)? |

Environment: Apple M2 Pro, 16 GB · Ollama 0.40.16 · `qwen2.5:7b-instruct` (Q4_K_M) · `nomic-embed-text` · 2026-11-20.

---

## `tokenizer_ratios.py`

| Content | chars/token |
|---|---|
| English prose | 6.54 |
| Code | 3.35 |
| Markdown table | 2.29 |
| Chinese | 1.44 |

A 4.5× spread, so `chars ÷ 4` is not a safe estimate. Numbers are tokenized digit by digit.

---

## `ollama_smoke.py`

Run: `python experiments/ollama_smoke.py [truncation cache_prefix show chat stream embed overhead default_ctx num_ctx overflow cache decode reload]`

### 1. Default context window is 4,096, not 6,144
A 5,057-token prompt sent **without** `num_ctx` → `/api/ps` shows context 4,096, `prompt_eval_count = 2,050`. The secret word at the start of the prompt was lost, and the model answered `'Pears'`. No error was returned.
→ **`num_ctx` must be sent on every request.**

### 2. `num_ctx=6144` works
The same prompt with `num_ctx=6144` → `prompt_eval_count = 5,088`, and the model answered `'BLUEBERRY'` ✅.

### 3. Overflow is silent and drastic: Ollama keeps only about **half** the window
| Sent (our count) | num_ctx | prompt_eval_count kept |
|---|---|---|
| 5,057 | 4,096 (default) | 2,050 |
| 7,074 | 6,144 | 3,074 |

In the overflow test, both the system prompt ("answer in ALL CAPS") and the secret were lost, and the answer was garbage: `'ApplesPearsInspectionDayCratesDistrictRecordSequence'`. **HTTP 200, no error.**

**Follow-up (`truncation`): it keeps half the *window*, not half the *prompt*, and it keeps the END.** Four code words were placed at 0%, 35%, 65% and 95% of the prompt:

| Sent | Kept | Half the prompt (hyp. A) | Half the window (hyp. B) | Code words that survived |
|---|---|---|---|---|
| 6,073 (fits) | 6,102 | — | — | APPLE, RIVER, CANDLE, TIGER |
| 6,362 | **3,074** | 3,181 | 3,072 | CANDLE, TIGER |
| 7,095 | **3,074** | 3,547 | 3,072 | CANDLE, TIGER |
| 9,066 | **3,074** | 4,533 | 3,072 | TIGER |
| 12,092 | **3,074** | 6,046 | 3,072 | TIGER |

Hypothesis B fits every row: **whenever the prompt is too long, Ollama keeps the last ~num_ctx/2 tokens**, however small or large the overflow. The surviving code words show the **end** is kept and the start dropped. At 9,066 tokens only the last 34% survives, so CANDLE (65%) is just lost. Overflowing by 218 tokens (6,362) still threw away **3,288 tokens**.

**Edge case: the prompt fits but the answer doesn't.** A 6,022-token prompt (122 tokens of room) with `num_predict=300` generated all **300** tokens, `done_reason=length`, no error. 6,022 + 300 > 6,144, so Ollama must have evicted early tokens *during generation* (context shifting) to keep going. The model loses the start of its prompt, where the system prompt sits, mid-answer, silently.
→ The answer reserve is **required, not optional**: prompt ≤ 6,144 − `num_predict` guarantees generation never runs out of room.

→ Prevention (exact budgeting) is essential, and detection (`prompt_eval_count` < expected) works.

### 4. Our tokenizer matches Ollama's exactly, once template overhead is added
| Messages | Our content count | prompt_eval_count | Overhead |
|---|---|---|---|
| 2 | 24 | 37 | 13 |
| 4 | 37 | 60 | 23 |
| 6 | 52 | 85 | 33 |

The overhead is exactly **5 tokens per message + 3** (the trailing `<|im_start|>assistant\n`). Formula: `prompt_tokens = Σ count(content) + 5 × n_messages + 3`.

**Without a system message**, Qwen's template injects its own default system prompt (+21 tokens). → Always send our own system message.

### 5. Prompt caching: `prompt_eval_count` always reports the FULL prompt ✅
| Request | prompt_eval_count | prompt_eval_cached_count | prefill time |
|---|---|---|---|
| ~1,550 tokens, cold | 1,558 | 3 | 5.37 s |
| Same prefix, different question | 1,557 | 1,549 | 0.22 s |
| ~3,050 tokens, cold | 3,053 | — | 10.82 s |
| Identical repeat | 3,053 | — | 0.03 s |

The caveat in D-010 is **not** an issue in this version. `prompt_eval_count` is the full prompt, and the cached part is reported separately in `prompt_eval_cached_count`. The truncation check is reliable.

**Bonus:** prefix caching is very effective (10.8 s → 0.03 s). Prompts that share a **prefix** with the previous request skip most of the prefill. This matters for how we order the prompt (see "Design implications").

**Follow-up (`cache_prefix`): reuse stops at the first changed token, and a cold control confirms it's the cache.** A ~3,056-token prompt, with a one-word change (`apples` → `plums`) at different depths:

| Run | Total | Cached | Prefill |
|---|---|---|---|
| 1 cold | 3,056 | 0 | 10.76 s |
| 2 identical | 3,056 | 3,055 | 0.03 s |
| 3 one word changed at 75% | 3,057 | **2,301 (75%)** | 2.83 s |
| 4 one word changed at 25% | 3,057 | **787 (26%)** | 8.09 s |
| 5 brand-new prefix (control) | 3,055 | 27 | **10.64 s** |

- The cached share equals the depth of the change, and prefill time is proportional to the *uncached* part. Everything after the first difference is recomputed, even though the rest of the text is identical.
- The control is slow again (10.64 s ≈ run 1), so the speed-ups come from the cache, not from warm-up.
- The control's 27 cached tokens are Qwen's injected default system prompt plus the template header, the only part identical to the previous request. That's the prefix rule at token level.

### 6. Speed
| Measurement | Result |
|---|---|
| **Prefill** (cache-cold) | **~275–285 tok/s** → a 5K-token prompt takes **~18.6 s before the first token** |
| Decode, short context | 33–38 tok/s → a 300-token answer takes ~9 s |
| Decode, ~5K context | ~23–34 tok/s (short samples, noisy) |
| Cold model load (first request) | 4.6 s |
| Time to first token, tiny prompt, warm model | 0.23 s |

The estimate in our design (~10 s for a 5K prefill) was **too optimistic by about 2×**.

### 7. Changing `num_ctx` reloads the model
`6144 → 6144`: 0 s · `6144 → 4096`: **4.4 s** · `4096 → 6144`: **2.8 s**. → `NUM_CTX` lives in one config constant.

### 8. Embeddings (nomic-embed-text)
- Vectors are 768-dim and come back **already L2-normalized** (norm 1.0000).
- **Input limit in Ollama: 2,048 nomic tokens, silently truncated** (a ~3,000-token input reported `prompt_eval_count=2048`, HTTP 200). Our ~400-token chunks are far below that.
- nomic uses its own (BERT) tokenizer, so counts differ from Qwen's. For example, 527 Qwen tokens = 406 nomic tokens on our filler text.
- Throughput: 64 chunks of ~140 tokens in 1.15 s ≈ **56 chunks/s**.
- Prefixes on the 3-document toy example: with prefixes, scores were `0.670 / 0.456 / 0.443`; without, `0.631 / 0.410 / 0.406`. The ranking was the same and the gap similar. **Toy example only; not evidence either way.** We keep the prefixes as the model card specifies, and the eval can test it.
- Real cosine scores sit in a **narrow band** (0.44–0.67 here, not 0–1), which confirms that a relevance floor must be calibrated, not guessed (D-015).

---

## Design implications (decided 2026-11-21)

1. **Time to first token is the demo's weak spot.** Full context ≈ 18 s of silence. The relevance filter (D-015) now matters for user experience. *Decision:* the context ceiling stays at ~3.7K; the relevance filter keeps typical prompts short.
2. **Prompt order affects the cache.** *Adopted* (D-011): `system → history → (context + question)`, so system + history form a cached prefix across turns.
3. **The margin can be small.** *Adopted* (D-011): the margin goes from ~200 to ~50, because counts are exact with the `5n + 3` formula.
4. **Warm the model up at chat startup.** It avoids a 4.6 s load on the first question (roadmap M3).
