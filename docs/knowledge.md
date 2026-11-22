# Knowledge base

Q&A collected during pairing sessions. Each entry answers one question, with concrete examples. Numbers marked *(illustrative)* are approximations, not measurements from this repo.

## Contents

1. [What does quantization `Q4_K_M` mean?](#1-what-does-quantization-q4_k_m-mean)
2. [What is `r.iter_lines()`?](#2-what-is-riter_lines)
3. [If we already have embeddings, why do we need a tokenizer?](#3-if-we-already-have-embeddings-why-do-we-need-a-tokenizer)
4. [Why does `exp_template_overhead` need `nonce()`?](#4-why-does-exp_template_overhead-need-nonce)
5. [When does Ollama's prompt cache clear or refresh?](#5-when-does-ollamas-prompt-cache-clear-or-refresh)
6. [How does our token count map to the 6,144 window? (budget, `5*messages + 3`)](#6-how-does-our-token-count-map-to-the-6144-window)
7. [How are messages wrapped in the ChatML template? (worked example)](#7-how-are-messages-wrapped-in-the-chatml-template)

---

## 1. What does quantization `Q4_K_M` mean?

**Where it shows up:** `qwen2.5:7b-instruct` (Q4_K_M) in [D-006](decisions.md) and [experiments/README.md](../experiments/README.md).

### Short answer

A model is a huge list of numbers (weights). **Quantization** stores each weight with fewer bits so the model is smaller and faster. `Q4_K_M` is the name of one specific recipe in the GGUF/llama.cpp format:

| Part | Meaning |
|------|---------|
| `Q4` | Weights are stored with about **4 bits** each (instead of 16). |
| `K`  | The **"K-quant"** scheme: weights are grouped into blocks, and each block gets its own scale (and min), so the 4-bit values are spread over the right range for that block. |
| `M`  | **Medium** variant. K-quants come as S (small), M (medium) and L (large). M is a mix: most tensors use 4-bit, but a few sensitive ones (parts of attention and feed-forward layers) use higher precision, 6-bit. |

### Why it exists: the memory math

A 7B model has about 7.6 billion weights *(Qwen2.5-7B is ~7.6B)*.

- fp16 (16 bits = 2 bytes per weight): 7.6e9 × 2 B ≈ **15 GB**. This does not fit a 16 GB M2 Pro once macOS, the KV cache and the embedding model take their share.
- Q4_K_M (about 4.8 bits per weight on average, because of the scales and the 6-bit tensors): 7.6e9 × 4.8 / 8 ≈ **4.6 GB**. It fits comfortably and loads fast.

That is the "about 3.5× smaller" claim in D-006.

### How block quantization works (worked example, illustrative)

Take 8 weights from one block (real blocks have 32 weights, grouped into super-blocks of 256):

```
original (fp16):   [ 0.12, -0.50,  0.33,  0.90, -0.07,  0.41, -0.88,  0.05 ]
```

1. Find the block's range: min = -0.88, max = 0.90.
2. 4 bits gives 16 levels (0 to 15). Scale = (0.90 - (-0.88)) / 15 ≈ 0.119.
3. Store each weight as an integer: `q = round((w - min) / scale)`.

```
w=0.12  -> (0.12+0.88)/0.119 = 8.4  -> 8
w=-0.50 -> 3.2  -> 3
w=0.33  -> 10.2 -> 10
w=0.90  -> 15
...
stored: [8, 3, 10, 15, 6, 11, 0, 8]   (4 bits each) + scale + min for the block
```

4. At inference, the weights are rebuilt: `w' = q × scale + min`.
   - 8 → 8 × 0.119 − 0.88 = 0.072. The original was 0.12, so there is a **rounding error** of about 0.05.

That error is the "quality loss". With billions of weights the errors mostly average out, so a 7B model at Q4 stays close to its fp16 version. The per-block scale is what keeps the error small: one block with large weights doesn't force a coarse scale onto another block with small weights.

### Why `K` and `M` matter

- **Older Q4_0 / Q4_1** use a simple per-block scale. **K-quants** (Q4_K) use a two-level scheme (a scale for each 32-weight block, itself quantized against a super-block scale), which gives better quality at similar size.
- **M (mixed)**: not every tensor tolerates 4 bits equally. Some attention (`attn_v`) and feed-forward (`ffn_down`) tensors hurt quality most when rounded, so Q4_K_M keeps them at 6 bits. That costs a little size and buys a noticeable quality gain. Q4_K_S skips the upgrade (smaller, slightly worse).

### Common alternatives

| Name | ~bits/weight | 7B size *(illustrative)* | Trade-off |
|------|--------------|--------------------------|-----------|
| fp16 | 16 | ~15 GB | Best quality, won't fit comfortably |
| Q8_0 | ~8.5 | ~8 GB | Near-lossless, double the memory |
| Q5_K_M | ~5.7 | ~5.4 GB | A bit better than Q4, a bit bigger |
| **Q4_K_M** | ~4.8 | **~4.6 GB** | The usual sweet spot |
| Q3_K_M | ~3.9 | ~3.7 GB | Noticeable quality drop |
| Q2_K | ~3.0 | ~3 GB | Often visibly degraded |

### What quantization does NOT change

- **The 6K context window** (D-002 and the token budget): that limit is about tokens in the prompt plus answer, not weight precision. Quantized weights do not shrink the budget.
- **The tokenizer**: token counts from the Qwen tokenizer are the same at any quantization.
- **The KV cache** (the memory that grows with context length) is stored separately, usually in fp16, so it is not reduced by Q4_K_M weight quantization.

### Failure modes to know

- **Quality loss shows up first in** precise arithmetic, rare facts, code and non-English text, and strict instruction-following. For a RAG app that answers from retrieved chunks, this is mostly fine, because the facts come from the context and not from the weights.
- **Lower than 4-bit** degrades quickly. Going from Q4 to Q3 or Q2 saves memory but costs more quality than going from fp16 to Q4.
- **Speed**: on Apple Silicon, token generation is memory-bandwidth bound, so fewer bytes per weight also means faster tokens. Quantization helps speed, not just size.

### Likely reviewer questions

- *"Why 4-bit? Doesn't it hurt quality?"* It's a small, measured loss at 7B. fp16 doesn't fit our 16 GB machine with headroom, and Q8 doubles the memory for a gain we'd struggle to see in RAG answers.
- *"What does the M mean?"* Medium mix: most tensors are 4-bit, the most sensitive ones are 6-bit.
- *"How can you verify which quantization you're running?"* `ollama show qwen2.5:7b-instruct` shows `quantization: Q4_K_M`, and [experiments/ollama_smoke.py](../experiments/ollama_smoke.py) prints `details.quantization_level`.

### Caveat

The bits-per-weight and size figures are approximate and from general llama.cpp knowledge, not measured here. If you want exact numbers for the demo, check the model size with `ollama list`.

---

## 2. What is `r.iter_lines()`?

**Where it shows up:** `for i, line in enumerate(r.iter_lines())` in `exp_stream()` in [experiments/ollama_smoke.py](../experiments/ollama_smoke.py).

### Short answer

`r.iter_lines()` returns an **iterator** (in `requests` it is a generator) that yields the HTTP response body **one line at a time, as the bytes arrive**. A `for` loop over it behaves like a `while` loop that keeps asking "give me the next line" until the server closes the stream.

The `for` loop is the same as this `while` loop:

```python
it = iter(r.iter_lines())
while True:
    try:
        line = next(it)      # blocks until the next full line arrives
    except StopIteration:    # server closed the connection
        break
    ...loop body...
```

The refinement is that the loop **waits** inside `next(it)`. It does not spin. It sleeps until the server sends more data. That waiting is what makes streaming work.

### Why `stream=True` is needed

```python
requests.post(url, json=body, stream=True)   # headers arrive, body is NOT downloaded yet
```

- `stream=False` (the default): `requests` downloads the whole body before returning, so you would get all tokens at once after the model finishes.
- `stream=True`: `requests` returns as soon as the headers arrive and leaves the body on the socket. `iter_lines()` then reads from the socket lazily.

### Trace (illustrative timing)

Ollama with `"stream": true` sends **NDJSON**: one JSON object per line, one line per generated chunk of text.

```
t=0.00s  request sent
t=0.40s  line 0: {"message":{"content":"1"},"done":false}      <- loop body runs, prints
t=0.45s  line 1: {"message":{"content":","},"done":false}
t=0.50s  line 2: {"message":{"content":" 2"},"done":false}
...
t=1.10s  line N: {"done":true,"eval_count":14,...}            <- final line carries stats
t=1.10s  server closes -> iterator ends -> loop exits
```

Between lines the loop is blocked in `next()`. The first line arriving at about 0.40 s is the "time to first token" that the script measures.

### Details worth knowing

- Each `line` is **bytes**, not `str`. That is why the script calls `json.loads(line)` (which accepts bytes) and `line.decode()` before printing.
- `enumerate(...)` just adds a counter `i`. It does not change how the iterator works.
- `iter_lines()` can yield **empty lines** (keep-alives). With Ollama's NDJSON this doesn't occur, but a general client would skip them with `if not line: continue`.
- The `with requests.post(...) as r:` block closes the connection when the loop ends, including when you break out early. That frees the server to stop generating.
- The iterator is **single-use**. Once exhausted, you can't loop over it again.

### Why it matters for the real app

The CLI needs to print tokens as they are generated (decision: CLI with streaming). The same pattern will be used there: loop over `iter_lines()`, parse each JSON line, print `message.content` immediately, and read the stats from the line with `done: true`.

### Likely reviewer question

- *"How does streaming work?"* The request sets `stream: true`, the server sends newline-delimited JSON as tokens are produced, and the client reads the open connection line by line with an iterator, so the first token shows up after about the prefill time and not after the whole answer is done.

---

## 3. If we already have embeddings, why do we need a tokenizer?

**Where it shows up:** [D-010](decisions.md) (token counting), D-002 (silent truncation), [experiments/tokenizer_ratios.py](../experiments/tokenizer_ratios.py).

### Short answer

They solve two different problems and use two different models:

| | Embedding model (`nomic-embed-text`) | Qwen tokenizer |
|---|---|---|
| Question it answers | "Which chunks are **about** the same thing as the query?" | "How much **room** do these chunks take in the chat model's 6,144-token window?" |
| Output | A vector (768 floats) per text | An integer: the token count |
| Used for | **Retrieval** (finding chunks) | **Budgeting** (deciding how many chunks, history turns and system prompt fit) |
| Belongs to | The embedding model | The chat model (`qwen2.5`) |

Embedding tells you *what to pick*. The tokenizer tells you *how much you can pick*. Retrieval alone can't stop an overflow.

### Worked example (illustrative numbers)

The budget is 6,144 tokens. Reserve 1,024 for the answer, which leaves 5,120 for the prompt.

```
system prompt          ~250 tokens
chat history (window)  ~600 tokens
user question           ~20 tokens
-----------------------------------
fixed part             ~870  ->  room for chunks = 5,120 - 870 = 4,250
```

Embedding search ranks 10 chunks by similarity: `C1 > C2 > C3 > ... > C10`. How many go in? That depends on each chunk's size **in Qwen tokens**:

```
C1 (prose)           390 tokens  -> running total  390
C2 (Markdown table)  610 tokens  -> 1,000
C3 (prose)           410 tokens  -> 1,410
...
```

Only exact counts let you keep adding chunks until the next one would exceed 4,250, then stop. Cosine similarity says nothing about size.

### Why a guess isn't good enough

Ollama has no overflow error. When the prompt is too long it **silently drops tokens from the start** (D-002; measured: 7,074 sent, 3,074 kept), and the start is where the system prompt sits. The answer then comes back confidently wrong, with no error.

Token counts vary a lot by content (measured, Qwen2.5 tokenizer):

| Content | chars/token |
|---|---|
| English prose | 6.54 |
| Code | 3.35 |
| Markdown table | 2.29 |
| Chinese | 1.44 |

A 4.5× spread, so the `chars ÷ 4` rule of thumb under-counts tables by about 1.7× and Chinese by about 2.8×, and the prompt overflows.

### "Can't the embed call give me the count?"

Not a useful one. Embedding models have **their own tokenizer** (nomic uses a BERT-style WordPiece vocabulary, Qwen uses byte-level BPE), so the same text gives different counts. The count that matters is the one the **chat model** will see, so it has to come from Qwen's tokenizer.

### Where else the tokenizer is used

- **Chunking** (~400-token target chunks): chunk size is measured in Qwen tokens, because the chunks are going into the Qwen prompt.
- **Pre-send guarantee:** the prompt builder never exceeds `6,144 - 1,024` (tested with pytest).
- **Drift detection:** after each response, compare our count with Ollama's `prompt_eval_count`. If Ollama processed fewer tokens, it truncated. If the two disagree persistently, the tokenizer file doesn't match the running model.

### Likely reviewer questions

- *"Why not just count characters or words?"* Measured 4.5× spread between content types. An estimate either wastes context or overflows it silently.
- *"Is downloading a tokenizer still 'fully local'?"* Yes: a static file fetched once at setup, like the model weights. Nothing goes over the network at runtime.
- *"What happens if you swap the chat model?"* The tokenizer must change with it. The config pairs them, and the drift check catches a mismatch.

---

## 4. Why does `exp_template_overhead` need `nonce()`?

**Where it shows up:** `nonce()` and `exp_template_overhead()` in [experiments/ollama_smoke.py](../experiments/ollama_smoke.py).

### Short answer

Ollama keeps a **prefix cache**: if a new request starts with the same tokens as the previous one, it skips re-processing that shared part. The experiment needs each request to be processed **from scratch** ("cache-cold"), so that timing and behaviour are comparable across runs. `nonce()` puts a unique string at the **start** of the prompt, so no request can share a prefix with an earlier one.

> **Correction (2026-11-21):** an earlier version of this entry said a cache hit makes `prompt_eval_count` under-report, which would corrupt the overhead number. That is **not true for this Ollama version (0.40.16)**. [experiments/README.md](../experiments/README.md) §5 measured it: `prompt_eval_count` is always the **full** prompt, and the cached part is reported separately in `prompt_eval_cached_count`. So the overhead formula would still come out right without a nonce. The nonce is still good practice (it makes every run cold and reproducible, and it keeps cached-token counts out of the other experiments' timings), but it is not what protects `exp_template_overhead`'s arithmetic.

### What the experiment measures

The chat template wraps every message in special tokens, e.g. `<|im_start|>user\n ... <|im_end|>\n`. The experiment sends 1, 3 and 5 messages and computes:

```
overhead = prompt_eval_count (Ollama)  -  sum of our count of each message's content
```

That gives the extra tokens per message, which the budget code must add on top of content counts.

### What would happen without it

All three runs start with the same system message and the same first turns, so runs 2 and 3 would reuse most of run 1's work:

```
run 1:  [system][user1]                              prompt_eval_count = 38   cached = 3
run 2:  [system][user1][asst1][user2]                prompt_eval_count = 69   cached = 38  (illustrative)
```

`prompt_eval_count` stays the full length, so `overhead` is unaffected. What changes is **speed and `prompt_eval_cached_count`**, which is why cold, reproducible runs matter for the timing experiments (`exp_cache`, `exp_cache_prefix`, prefill speed).

### How `nonce()` fixes it

```python
def nonce():
    return f"[run {uuid.uuid4().hex[:8]}]"      # e.g. "[run 3f9a01bc]"
```

- It is **called again inside the loop** (`f"{nonce()} You answer from context only."`), so each run gets a different first line: `[run 3f9a01bc] ...`, `[run 8d22e4a7] ...`, and so on.
- Prefix matching works token by token from the start, so a different **first** token means nothing can be reused. If the nonce were at the end, the long shared beginning would still be cached.
- It also stops a re-run of the script a minute later from hitting the cache left by the previous run.

### Why the nonce doesn't distort the measurement

The nonce is part of the message content, so `count(m["content"])` includes its tokens. They appear on both sides of the subtraction and cancel out of `overhead`.

### Same trick elsewhere

Every experiment that reads timing or token counts uses `nonce()` for the same reason (`exp_cache` uses it once so the "1st" request is truly cold, then repeats the identical prefix on purpose to show the cache).

### Why it matters for the real app

The D-010 truncation check ("compare `prompt_eval_count` with our expected count") **is reliable** on this version, because `prompt_eval_count` is the full prompt. Keep the caveat in the notes: on a different Ollama version, re-run `exp_cache` to confirm this still holds.

### Likely reviewer question

- *"How did you measure the template overhead?"* Compared our exact content count with Ollama's `prompt_eval_count`. Result: 5 tokens per message + 3. Requests were cache-cold, using a unique nonce at the start of each prompt.

---

## 5. When does Ollama's prompt cache clear or refresh?

**Where it shows up:** `exp_cache_prefix()` in [experiments/ollama_smoke.py](../experiments/ollama_smoke.py), results in [experiments/README.md](../experiments/README.md) §5.

### What the cache actually is

It is the model's **KV cache** (the key/value tensors computed for each prompt token during prefill), kept in memory by the running model process. It is not a separate cache file or a database. "Reuse" means: compare the new prompt's tokens with the tokens whose KV values are already in memory, find the longest **common prefix**, and only compute the rest.

### What the experiment shows (measured, README §5)

| Run | What was sent | Cached | Why |
|---|---|---|---|
| 1 | cold prompt | 0 | nothing in memory |
| 2 | identical | 3,055 | whole prompt is a prefix match |
| 3 | one word changed at 75% | 2,301 (75%) | match stops at the first differing token |
| 4 | one word changed at 25% | 787 (26%) | same rule, earlier cut |
| 5 | brand-new prefix | 27 | only the template header matched |

Everything after the first different token is recomputed, even if the text after it is identical.

### Why does `exp_cache_prefix` still use `nonce()`?

The nonce is created **once** (`head = nonce()`) and shared by runs 1-4, so they *do* share a prefix. It is replaced once, before run 5.

```
run 1-4:  [run 3f9a01bc]\nRecord 0: ...    same head -> shared prefix
run 5:    [run 8d22e4a7]\nRecord 0: ...    new head  -> differs from token 1 (control)
```

Its two jobs: (1) make run 1 truly cold, because the deterministic `filler(3000)` text is identical on every script run and the server's cache outlives the script; (2) create the control (run 5), whose first token differs so nothing can match. Compare `exp_template_overhead`, which calls `nonce()` per request because every request there should be cold.

Rule: call `nonce()` once per group of requests that should share a cache, and again for a fresh start.

### When it refreshes (every request)

After each request, the cache holds **that request's** tokens. Tokens that were beyond the match point are replaced by the new ones. You can see this in the table: run 4 is compared with what run 3 left behind, and the match is 26% since run 3 and run 4 agree only up to the 25% line. The cache follows the **most recent prompt**, it does not accumulate a history of all earlier prompts. So alternating between two unrelated prompts gets no benefit.

### When it clears completely

*These come from Ollama's documented behaviour and general llama.cpp design. Only the first is something we measured, and only in the sense that `reload` exists as an experiment: check the others if the demo depends on them.*

- **The model is unloaded.** Ollama unloads an idle model after `keep_alive` (default **5 minutes**), freeing the memory, and the KV cache goes with it. The next request is cold and also pays the model load time. (This is what `exp_reload` is about. Set `keep_alive` longer or to `-1` to hold it.)
- **The model is reloaded with different settings.** Changing `num_ctx` (or switching to another model) makes Ollama restart the runner, which empties the cache. This is another reason to send the **same** `num_ctx: 6144` on every request.
- **Ollama restarts or the machine sleeps/reboots.** Nothing persists across process restarts.
- **Concurrent requests.** With parallel slots, different conversations may land in different slots, each with its own cache. We run a single CLI user, so this should not matter.

### What does NOT clear it

- Time between requests, as long as the model is still loaded.
- Your Python process exiting: the cache lives in the Ollama server, not in the client. (This is why re-running the script soon after can hit leftover cache, and why `nonce()` exists.)

### Design implication for the app

Each chat turn should **start with the same text as the previous turn**, so Ollama can reuse its work. The planned order `system -> history -> (context + question)` makes system + history a shared prefix, and only the new chunks and question need prefill (about 18 s for 5K tokens cold, versus a fraction of a second for the cached part).

The catch: the retrieved chunks change every turn, so the cache helps only up to the first changed token. If the history is **trimmed from the front** by the sliding window, the prefix shifts and the cache is lost for that turn. That is a point to weigh when the history-window decision is implemented: dropping whole old turns in one step, not one by one, is cheaper.

### Likely reviewer question

- *"How does caching affect your prompt design?"* Put stable content first (system prompt, then history), volatile content last (retrieved chunks, question). Measured: a one-word change at 25% depth reuses only 26% of the prompt, so the position of the first change decides the savings.

---

## 6. How does our token count map to the 6,144 window?

**Where it shows up:** token budget table in [decisions.md](decisions.md) (D-011), D-010 (template overhead), [experiments/README.md](../experiments/README.md) §3-4.

### Step 1: what 6,144 actually limits

`num_ctx = 6144` is a limit on **prompt tokens + generated tokens together**. They live in the same window:

```
prompt_tokens + generated_tokens  <=  6,144
```

We can't know how long the answer will be, so we cap it with `num_predict = 1024` and reserve that room up front:

```
prompt_tokens  <=  6,144 - 1,024  =  5,120
```

That is where "5,120 for the prompt" comes from. The 1,024 is a ceiling on the answer, not a prediction. (Measured: a 6,022-token prompt with `num_predict=300` still generated 300 tokens, because Ollama silently evicted the start of the prompt mid-answer to make room. The reserve is what prevents that.)

### Step 2: what `prompt_tokens` is, exactly

Ollama doesn't feed raw text to the model. It wraps every message in Qwen's chat template (ChatML):

```
<|im_start|>system\nYou answer from context only.<|im_end|>\n
<|im_start|>user\nHow do I turn on sync?<|im_end|>\n
<|im_start|>assistant\n                      <- header for the answer to come
```

The special tokens and newlines around each message add **5 tokens per message** (`<|im_start|>`, `role`, `\n`, `<|im_end|>`, `\n`), and the trailing `<|im_start|>assistant\n` adds **3**. Measured to match Ollama exactly:

```
prompt_tokens = sum(count(content of each message)) + 5 * n_messages + 3
```

Why this matters: our tokenizer counts only the *text* we give it. If we budget with `sum(count(content))` alone, we under-count by 5 per message, and the gap grows with the number of messages. The formula closes the gap, so our count equals what Ollama will see.

### Step 3: worked example (content counts are illustrative)

A turn with 2 earlier exchanges in the history: messages = system + 4 history + 1 final user message (retrieved chunks + question) = **6 messages**.

```
system content                     280
history content (4 messages)       400
final user: chunks 2,100 + question 18   2,118
---------------------------------------------
sum of content                   2,798
template overhead: 5 * 6 + 3        33
---------------------------------------------
prompt_tokens                    2,831     <= 5,120  OK  (2,289 spare)
```

Overhead grows with conversation length: with 10 history messages it would be 5 * 12 + 3 = **63**. Small, but not zero, and it only stays correct if we count it.

### Step 4: how the budget table's numbers connect

```
6,144  window
- 1,024  answer reserve (num_predict)
-   300  system prompt
- 1,024  history (cap)
-    50  safety margin
= 3,746  retrieved-context ceiling
```

(You wrote 3,726 earlier. The doc says **3,746**.) Each line is a **ceiling**, not a target (D-011). The table answers: "if every slot were full, what is the most the chunks could take?" The relevance filter (D-015) usually sends far less.

### Where `5 * n + 3` lives in that table (honest gap)

The table doesn't say which slot pays the template overhead. D-009 says history counts its own overhead inside its 1,024. For the rest, here is the full-slots arithmetic:

```
system:   300 content + 5            = 305
history:  1,024 (overhead included)  = 1,024
final user message: +5, trailing +3  =   8
question tokens (not a row in the table) ~  20
chunks:   3,746
answer:   1,024
-----------------------------------------------
                                       6,127   vs 6,144 -> only 17 left of the 50 margin
```

So when every slot is full, the margin is already being eaten by the overhead and the question, which aren't listed in the table. The system is still safe, but only because of the margin. The more robust way to do it:

**Compute the context budget at runtime instead of trusting a fixed 3,746.**

```
fixed  = count(system) + count(history incl. overhead) + count(question)
         + count(instructions/wrappers around chunks) + 5 * n_messages + 3
context_budget = min(3,746,  6,144 - num_predict - margin - fixed)
```

Then the builder adds chunks best-first until `context_budget` is used, counting each chunk's text **and** its formatting (e.g. a `[source: file.md]` header). The 3,746 stays as a cap, but the true limit is "whatever is left after the exact count of everything else". This is the safer design for the review, and a unit test can assert `prompt_tokens + 1,024 <= 6,144` for adversarial inputs (full history, long question, many chunks).

*This is a proposed refinement to D-011, not yet agreed. Raise it before the prompt builder is written.*

### Failure modes if this is wrong

- **Undercount (forgot overhead or the question):** prompt creeps over 5,120. `prompt + answer` then exceeds 6,144 and Ollama silently evicts the **start** of the prompt during generation, losing the system prompt, with no error.
- **Prompt itself over 6,144:** Ollama keeps only the last ~3,072 tokens (measured), so most context vanishes.
- **Overcount:** only wasted capacity, a safe failure.
- **Detection:** compare `prompt_eval_count` with our expected number after each response. They matched exactly in the measurements.

### Likely reviewer questions

- *"How do you guarantee the model never exceeds 6K?"* `num_ctx=6144` on every request, an exact count `sum(content) + 5n + 3` for the prompt, `num_predict=1024` reserved, and the prompt builder stops adding chunks when `prompt + 1,024` would exceed 6,144. A test asserts it.
- *"Where does `5n + 3` come from?"* Measured against `prompt_eval_count`: Qwen's ChatML template adds 5 tokens per message and 3 for the assistant header.
- *"Why is the answer reserve a fixed 1,024?"* Generated tokens share the window; we can't know the length in advance, so we cap it and reserve the cap.

---

## 7. How are messages wrapped in the ChatML template?

**Where it shows up:** D-010 ("ChatML template"), entry 6 above. Numbers below were produced with the project's Qwen2.5 tokenizer file (`models/qwen2.5-tokenizer.json`), not estimated.

### The wrapper

Each message becomes `<|im_start|>` + role + `\n` + content + `<|im_end|>` + `\n`. After the last message, Ollama appends an open `<|im_start|>assistant\n`, and the model continues from there.

### Example

Messages: system "You answer from context only.", user "Hello, this is Ryan", assistant "This model has 128k parameters.", user "What is my name?".

```
<|im_start|>system\nYou answer from context only.<|im_end|>\n
<|im_start|>user\nHello, this is Ryan<|im_end|>\n
<|im_start|>assistant\nThis model has 128k parameters.<|im_end|>\n
<|im_start|>user\nWhat is my name?<|im_end|>\n
<|im_start|>assistant\n
```

Tokens of one message (`Ġ` = leading space, `Ċ` = newline):

```
<|im_start|> user Ċ | Hello , Ġthis Ġis ĠRyan | <|im_end|> Ċ
   3 wrapper tokens        5 content tokens       2 wrapper tokens
```

`<|im_start|>` and `<|im_end|>` are single special tokens. The assistant message has 10 content tokens, because `128k` is split digit by digit: `Ġ`, `1`, `2`, `8`, `k`.

| Message | Content | With wrapper |
|---|---|---|
| system | 6 | 11 |
| user | 5 | 10 |
| assistant | 10 | 15 |
| user | 5 | 10 |
| trailer `<|im_start|>assistant\n` | | 3 |
| **Total** | **26** | **49** |

`26 + 5 x 4 messages + 3 = 49`, equal to the token count of the whole string. Counting content only would give 26, 23 short.

### Why this matters

- The 5 per message and 3 for the trailer are exactly the formula in entry 6.
- Without a system message, Qwen's template inserts its own default system prompt (+21 tokens measured), which is why we always send our own.
- The special tokens are real tokens in the model's vocabulary. User text containing the literal string `<|im_end|>` could in principle confuse the template; worth a thought if the app ever accepts untrusted text, though our documents are local.
