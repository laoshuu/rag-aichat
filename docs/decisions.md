# Design Decisions

A log of the architectural decisions behind this project, kept as prep for the code review.

**Status legend**
- `accepted`: the current design. Decisions stay **open to change while building**. If implementation shows a better option, update the entry (or supersede it) and note what we learned.
- `superseded by D-XXX`: replaced by a later decision.

Facts marked **[verify]** come from documentation or memory and have not yet been checked against the real system. Confirm each one when we build that component.

---

## Architecture at a glance

```
                     INGEST (offline, `rag-chat ingest <dir>`)
 .md/.txt files ──► loader ───► chunker ──► embedder ──────► index/ (vectors.npy + chunks.json)
                                (heading-aware,  (Ollama:
                                 ~400 tok)        nomic-embed-text)

                     CHAT (interactive, `rag-chat chat`)
 user question ─┬─► q = 0.8·embed(current) + 0.2·embed(prev) ─► NumPy cosine search
                │                                                       │ ranked chunks
                │                                                       ▼
                │   history (sliding window) ──────────────► prompt builder (6,144-token budget,
                │                                             exact counts via Qwen tokenizer)
                │                                                       │
                ▼                                                       ▼
          CLI (streams answer, prints sources) ◄──────── Ollama: qwen2.5:7b-instruct (num_ctx=6144)
```

### Token budget (D-011)

| Slot | Tokens | Notes |
|---|---|---|
| Answer (generation) | 1,024 | Enforced by `num_predict`. Generated tokens share the context window with the prompt. |
| System prompt | ~300 | Fixed text, measured once. |
| Chat history | ≤ 1,024 | Sliding window (D-009). |
| Safety margin | ~50 | Counting is exact (`Σ content + 5·messages + 3`, measured), so this only covers small edits such as a longer system prompt. Reduced from ~200 on 2026-11-21. |
| **Retrieved context + question** | **≤ 3,746** | Whatever is left. The question comes out of this slot; chunks fill the rest best-first, relevant ones only (D-015). This is a ceiling, not a target. |
| **Hard limit (prompt)** | **≤ 5,120** | `6,144 − 1,024`. Enforced by `prompt.py` on the exact message count. The rows above are how we allocate; this is what we guarantee. |
| **Total** | **6,144** | Sent as `num_ctx` on every request. |

---

## Foundations

### D-001: Language: Python
- **Status:** accepted
- **Context:** We need first-class bindings for local LLM runtimes, embedding models and vector stores.
- **Options considered:**
  - **Python.** Pros: every RAG building block has a mature Python library (Ollama client, llama-cpp-python, sentence-transformers, NumPy, FAISS); it's the ML lingua franca, so reviewers will expect it; fast to iterate in. Cons: slower than compiled languages, and dependency management takes some care.
  - **TypeScript/Node.** Pros: well suited to web UIs, has an Ollama client. Cons: weaker numeric and ML tooling (no NumPy equivalent), and tokenizer libraries are less mature.
  - **Go/Rust.** Pros: fast, ship as a single binary. Cons: few RAG libraries, so we'd write more ourselves, and iteration is slower.
- **Decision:** Python 3.11+.
- **Why:** The bottleneck is LLM inference inside Ollama, not our glue code, so Python's speed doesn't matter here and its ecosystem does.
- **Trade-offs accepted:** We need a venv and pinned dependencies to stay reproducible (D-021).
- **Likely reviewer questions:**
  - *"Isn't Python slow?"* Our code mostly waits on I/O and does a NumPy matrix multiply. More than 95% of latency is token generation inside Ollama, which is C++/Metal.

### D-002: LLM runtime: Ollama
- **Status:** accepted (deep-dive 2026-11-19)
- **Key insight from the deep-dive:** Ollama and llama-cpp-python run the **same engine** (llama.cpp/ggml on Metal) with essentially the same speed and quality. The choice is about **how much of the stack around the engine we own**, not about performance. Ollama takes on model loading, memory management, crash isolation, installation and downloads, and keeps the model in memory across app restarts (a faster development loop). The price is silent truncation and the lack of a built-in tokenizer, both handled in D-010.
- **Context:** The LLM must run locally on an Apple M2 Pro with 16 GB of unified memory. Most local runtimes run quantized **GGUF** models through **llama.cpp**, which uses the Apple GPU via Metal. The real question is how our app talks to llama.cpp.
- **Options considered:**
  - **Ollama.** A local server wrapping llama.cpp, with a REST API on `localhost:11434`.
    - Pros: one-command install. `ollama pull <model>` handles downloads, quantization variants and chat templates. Metal acceleration is automatic. The model runs as a **separate process**, so the app is a thin HTTP client and models swap with one config value. Easy for reviewers to reproduce. It also serves embeddings (D-007), so one runtime covers both.
    - Cons: a background service is one more moving part. **No public tokenize endpoint**, so we count tokens ourselves (D-010). Its **default context window is 4,096** (measured, Ollama 0.40.16), so we must send `num_ctx: 6144` with every request. When a prompt overflows, Ollama **truncates it silently** (HTTP 200, no error), **keeping only the last ~num_ctx/2 tokens** whatever the size of the overflow, so the start goes, which is where our system prompt sits. Measured at num_ctx = 6,144: prompts of 6,362 / 7,095 / 9,066 / 12,092 tokens all kept exactly 3,074. A prompt that fits but leaves no room for the answer is also silently context-shifted during generation (see `experiments/README.md` §3). We also give up some low-level control (sampling internals, KV-cache reuse).
  - **llama-cpp-python.** Python bindings that run llama.cpp **inside our process**.
    - Pros: `Llama(n_ctx=6144)` sets the window directly. **`llm.tokenize()` gives exact counts with the exact model tokenizer**, with no extra dependency. No separate service to run, and full control over sampling.
    - Cons: the pip install compiles C++ with Metal flags, which can fail on some machines (a demo risk; prebuilt Metal wheels exist but their availability varies; not tested, since we didn't choose this option). We download GGUF files by hand from Hugging Face and pick the quantization ourselves. Our app manages the model's lifecycle and memory, and the model loads on every app start (several seconds). Chat templates are more manual.
  - **llama.cpp's `llama-server` (raw).** The upstream C++ HTTP server, with an OpenAI-compatible API.
    - Pros: no wrapper, the most control, and it exposes a `/tokenize` endpoint.
    - Cons: you build or install it and manage GGUF files and flags yourself. It's essentially "Ollama without the conveniences".
  - **MLX / mlx-lm.** Apple's ML framework.
    - Pros: often the fastest option on Apple Silicon.
    - Cons: Mac-only, so reviewers on Linux or Windows can't run it. Smaller ecosystem.
  - **LM Studio.** A GUI app with an OpenAI-compatible local server.
    - Pros: very user-friendly.
    - Cons: GUI-driven, so hard to script and reproduce. Closed source.
- **Decision:** Ollama.
- **Why:** The most convenient option for development, and the lowest-risk path to a working demo for someone new to local LLMs. It separates model serving from application logic cleanly, and reviewers can reproduce it with `ollama pull`. Its two weaknesses (context size and token counting) are things we take on explicitly, and they make good review talking points.
- **Trade-offs accepted:**
  - **We own 6K enforcement.** `num_ctx` comes from a single config constant on every call, because a request with a different value makes Ollama reload the model. Our own exact counting, plus a test, guarantees the assembled prompt is always ≤ 6,144 − 1,024 tokens.
  - **Silent truncation.** Mitigated by the budget guard (prevention) and a `prompt_eval_count` check after each response (detection). See D-010.
  - **A separate tokenizer file** instead of llama-cpp-python's built-in `tokenize()`. Note that token counting itself is *not* a cost of Ollama: we need counts to budget chunks with any runtime.
  - Not a cost: speed. The HTTP hop takes milliseconds next to seconds of generation.
- **When llama-cpp-python would win:** if exact in-process control (built-in tokenizer, loud overflow errors, custom sampling or grammars) mattered more than convenient setup, or if we were shipping a single desktop app without a separate service.
- **Likely reviewer questions:**
  - *"The model supports 32K natively. How do you guarantee 6K?"* Two layers. (1) The runtime layer: `num_ctx=6144` on every request means the KV cache is physically 6K. (2) The application layer: the prompt builder counts tokens exactly and never assembles more than 6,144 − 1,024 (the answer reserve) tokens. A unit test asserts this.
  - *"What happens if you overflow?"* Ollama silently truncates, the system prompt gets dropped, and the model stops following grounding rules. That's why we enforce the limit *before* sending rather than relying on the runtime.
  - *"Why not llama.cpp directly?"* We'd gain exact tokenization in-process, but take on the compile risk and model-file management. Ollama is llama.cpp underneath, so performance is the same.

### D-003: No RAG framework; we write the pipeline ourselves
- **Status:** accepted
- **Context:** LangChain and LlamaIndex supply loaders, splitters, vector stores and prompt assembly.
- **Options considered:**
  - **Our own pipeline.** Pros: every step is visible and explainable; exact control over what enters the 6K prompt; few dependencies; easy to test. Cons: more code to write and own; we re-implement things like text splitting.
  - **LlamaIndex/LangChain.** Pros: least code, many integrations, quick prototypes. Cons: prompt assembly and truncation are abstracted away, so the 6K guarantee becomes "the framework handles it". Heavy dependency trees and frequent API churn. Harder to debug.
  - **Hybrid** (borrow a framework's splitter, write the rest). Pros: saves a little code. Cons: pulls in the whole dependency tree for a small piece.
- **Decision:** Our own pipeline.
- **Why:** In a code review, the interesting parts of a 6K-context RAG system *are* the parts a framework hides: chunking, budgeting and prompt assembly.
- **Trade-offs accepted:** Fewer features (no reranking or hybrid search out of the box), and the bugs in our splitter are ours.
- **Likely reviewer questions:**
  - *"Isn't this reinventing the wheel?"* The wheel here is a few hundred lines. In exchange we control the token budget exactly and can explain every line. For production at scale, I'd look again at a framework or a vector DB.

### D-004: Corpus scale: small
- **Status:** accepted
- **Context:** Corpus size drives the choice of vector store, ingestion design and performance work.
- **Options considered:** small demo corpus (tens of files, a few MB); reviewer-supplied files at demo time; large (thousands of files).
- **Decision:** Small: tens of files that we curate (see D-022).
- **Why:** That's what the demo needs. Designing for thousands of files would add an approximate-nearest-neighbour (ANN) index, incremental indexing and batching that we'd have to build and defend without needing them.
- **Trade-offs accepted:** Not tuned for large corpora, but the upgrade path is clear and explainable (D-008).
- **Likely reviewer questions:**
  - *"What if I drop in 10,000 files?"* Ingestion slows down linearly because embedding is the bottleneck, and brute-force search still works up to about 100K chunks. Beyond that we'd move to FAISS/HNSW and incremental indexing.

### D-005: Interface: CLI chat
- **Status:** accepted
- **Options considered:**
  - **CLI.** Pros: smallest; no web framework; easy to script and test. Cons: less visual polish.
  - **Streamlit/Gradio web UI.** Pros: chat bubbles, file uploads, looks polished. Cons: extra framework, and Streamlit's rerun-the-script model complicates state.
  - **FastAPI backend + client.** Pros: clean layering, easy to add a UI later. Cons: an extra layer to build before the core works.
- **Decision:** CLI with two commands, `ingest` and `chat`.
- **Why:** The review is about architecture, and a CLI keeps attention on the pipeline. Because the core is a library package (D-018), adding a web UI later just means writing another front end.
- **Likely reviewer questions:**
  - *"How would you add a web UI?"* The CLI is a thin layer over the package. A FastAPI or Streamlit front end would call the same retriever and prompt builder.

---

## Model layer

### D-006: Chat model: `qwen2.5:7b-instruct` (Q4_K_M)
- **Status:** accepted
- **Context:** The model must fit in 16 GB alongside the OS and the embedding model, run fast enough for a live demo, and stick to the provided context.
- **Memory math:** 7B parameters at ~4.5 bits each ≈ 4.7 GB of weights. KV cache at 6K tokens for Qwen2.5-7B (28 layers, 4 KV heads × 128 dims, fp16) ≈ 2 × 28 × 4 × 128 × 2 bytes ≈ 56 KB per token × 6,144 ≈ **~340 MB**. Total ≈ 5–5.5 GB, which fits comfortably.
- **Options considered:**
  - **qwen2.5:7b-instruct.** Pros: strong instruction-following and context use; Apache-2.0 licence; **no "thinking" mode**, so no hidden reasoning tokens eat the answer budget; its tokenizer is downloadable without a licence gate. Cons: not the newest model; it was trained with a default system prompt in its template, so we must always send our own.
  - **llama3.1:8b-instruct.** Pros: very well known. Cons: slightly larger; its Hugging Face tokenizer sits behind a licence click, which makes exact counting awkward (D-010).
  - **qwen3:8b.** Pros: newer and stronger at reasoning. Cons: "thinking" mode is on by default and emits long reasoning before answering, which consumes the 1,024-token answer reserve unless disabled.
  - **gemma3:4b.** Pros: about 3 GB and faster. Cons: weaker at strict grounding and at saying "I don't know".
  - Larger (12–14B). Pros: better quality. Cons: about 8–9 GB of weights on a 16 GB machine; slower generation and memory pressure in a live demo.
- **Decision:** `qwen2.5:7b-instruct`, configurable.
- **Why:** The best balance of grounding quality, speed (measured on the M2 Pro: decode 33–38 tok/s, prefill ~280 tok/s; native context 32,768, which we cap at 6,144) and a predictable token budget.
- **Trade-offs accepted:** The tokenizer is tied to the model family. Switching to a non-Qwen model means switching the tokenizer file too (D-010).
- **Likely reviewer questions:**
  - *"Why 4-bit quantization? Doesn't it hurt quality?"* Q4_K_M is the usual sweet spot. Quality loss compared with fp16 is small for 7B models, and it cuts memory by about 3.5×.
  - *"Why not a bigger model?"* Memory and latency on a 16 GB laptop. A 7B model with good retrieval beats a 14B model that stutters in a demo.

### D-007: Embeddings: `nomic-embed-text` via Ollama
- **Status:** accepted
- **Context:** Retrieval needs a local model that turns text into vectors whose cosine similarity reflects meaning.
- **Options considered:**
  - **nomic-embed-text (Ollama).** Pros: reuses the runtime we already run; no PyTorch (~2 GB) install; 768-dim vectors; input limit of **2,048 nomic tokens in Ollama, silently truncated** beyond that (measured), far above our ~400-token chunks. Vectors come back already L2-normalized; throughput ≈ 56 chunks/s. Cons: needs task prefixes (`search_document: ` when indexing, `search_query: ` when searching), and forgetting them quietly degrades retrieval. An HTTP call per batch.
  - **sentence-transformers (bge-small, all-MiniLM-L6-v2).** Pros: in-process, hugely popular, many model choices. Cons: PyTorch dependency; a second model-loading path; MiniLM truncates input at 256 tokens.
  - **mxbai-embed-large (Ollama).** Pros: stronger on benchmarks; 1024-dim. Cons: a **512-token input limit** caps chunk size, and it's larger and slower.
- **Decision:** `nomic-embed-text` through Ollama's embed API.
- **Why:** One runtime for both models keeps the stack small, and its input limit comfortably covers 400-token chunks plus heading prefixes.
- **Trade-offs accepted:** If we ever change the embedding model, the whole index must be rebuilt, because vectors from different models aren't comparable. We'll store the model name in the index metadata and refuse to load a mismatched index.
- **Implementation (M2):**
  - **We L2-normalize anyway**, even though Ollama's nomic vectors already come back with norm 1.0000 (measured in M0). It's one line. With it, "cosine = dot product" is guaranteed by our code, not by something we happened to observe about one model. Without it, switching `EMBED_MODEL` to a model that doesn't normalize would silently turn `M @ q` into a length-biased score.
  - Texts go to `/api/embed` in **batches of ~64**, with a progress line, so a few-hundred-chunk ingest doesn't look hung and every request stays small.
  - **Two first-run errors get a clear message and exit 1:**
    - Connection refused: "run `ollama serve`".
    - HTTP 404 `{"error":"model '…' not found"}` (verified): "run `ollama pull nomic-embed-text`".
  - **Tested without Ollama:** `requests.post` is monkeypatched to check that both prefixes are sent and that the output is normalized. Those are the two silent-failure modes.
- **Likely reviewer questions:**
  - *"Why different prefixes for queries and documents?"* nomic was trained with asymmetric task prefixes. A short question and a long passage get mapped into a shared space more accurately when the model knows which is which.
  - *"Why cosine similarity?"* It compares direction and ignores length. We L2-normalize vectors once at ingest, so cosine becomes a plain dot product.

### D-008: Vector store: NumPy brute-force
- **Status:** accepted
- **Options considered:**
  - **NumPy.** An `N×768` float32 matrix plus a parallel list of chunk metadata. Pros: exact search; ~30 lines; no extra dependencies; easy to reason about and test. Cons: O(N) per query; we write persistence ourselves; no metadata filtering.
  - **ChromaDB.** Pros: persistence, metadata filtering, a real DB API. Cons: a dependency plus an abstraction this scale doesn't need.
  - **FAISS.** Pros: industry-standard ANN with an obvious scaling story. Cons: a native dependency, and at our scale brute force is already exact and instant.
  - **SQLite + sqlite-vec.** Pros: a single file, SQL filtering. Cons: an extension to install, and still more machinery than we need.
- **Decision:** NumPy, exact cosine search over normalized vectors (`scores = M @ q`, then `argsort`).
- **Why:** At hundreds to thousands of chunks, a matrix multiply takes well under a millisecond. Exact results also make the retrieval eval (D-019) deterministic.
- **Trade-offs accepted:** Memory is N × 768 × 4 bytes (100K chunks ≈ 300 MB), and search grows linearly.
- **Interface (M2):**
  - `save(dir, chunks, vectors)` (builds `manifest.json` itself from config, so it always describes the run that produced the vectors), `load(dir) -> Index`, and `Index.search(q) -> [(chunk, score)]`.
  - `search` returns **all** chunks sorted best-first, not a top-k, because D-015 filters afterwards (floor → relative cutoff → budget) and shouldn't have to guess a k up front.
  - There's **no `add`**, because ingest always rebuilds the whole index (D-012), so nothing would ever call it.
- **Likely reviewer questions:**
  - *"How does this scale?"* Linearly. It's fine up to roughly 10⁵ chunks. Beyond that, FAISS IVF/HNSW trades a little recall for sub-linear search. The store sits behind a small interface (`save`, `load`, `search`), so swapping it is local to one module.

---

## Context budget (the 6K constraint)

### D-009: Chat history: token-bounded sliding window
- **Status:** accepted
- **Options considered:**
  - **Sliding window.** Keep the newest turns that fit in ≤ 1,024 tokens and drop the oldest first. Pros: simple, predictable, testable. Cons: older context is lost completely.
  - **Stateless.** Pros: maximum room for retrieved context. Cons: breaks follow-up questions, and this is billed as a chat app.
  - **Rolling summary.** Pros: keeps long-range memory. Cons: an extra LLM call per turn (slow on a laptop); summaries can drop or distort facts; another prompt to budget.
- **Decision:** Sliding window over whole (user, assistant) turn pairs, capped at 1,024 tokens, counting template overhead.
- **Why:** Short demo conversations fit easily, and the behaviour is deterministic and easy to explain.
- **Trade-offs accepted:** A long answer (up to 1,024 tokens) can push all earlier turns out of the window. If that shows up in testing, we can store a truncated version of long assistant answers in history. **Cache interaction:** history sits right after the system prompt (D-011 prompt order). When the window drops its oldest turn, the history changes near the start of the prompt, so that turn gets almost no prefix-cache benefit. Turns where nothing is dropped reuse system + history.
- **Likely reviewer questions:**
  - *"What if a single turn exceeds the history budget?"* It gets dropped, or truncated if we add that. History is the first thing to give way, because the documents are the point of RAG.

### D-010: Token counting: Qwen2.5's tokenizer, run locally
- **Status:** accepted (amended after the D-002 deep-dive)
- **Context:** We count tokens for **two separate reasons**:
  1. **Budgeting:** the prompt builder must know how many chunks fit (D-015). This applies with *any* runtime.
  2. **Overflow protection:** Ollama has no public tokenize endpoint and truncates silently (D-002), so we must guarantee the prompt fits *before* sending.
- **Options considered:**
  - **Exact.** Load Qwen2.5's `tokenizer.json` with the Hugging Face `tokenizers` library (a Rust core, lightweight, no PyTorch). Pros: exact counts. Cons: one dependency; a one-time download at setup (it then runs offline, like the model weights); tied to the model family.
  - **Heuristic (chars ÷ 4 + margin).** Pros: no dependencies. Cons: badly wrong for code, tables and non-English text (Chinese is roughly one token per character), which can push the prompt past 6K.
  - **Measure afterwards (`prompt_eval_count`).** Pros: ground truth from Ollama. Cons: only detects overflow after it happens.
- **Measured evidence (Qwen2.5 tokenizer, 2026-11-19):** prose 6.54 chars/token · code 3.35 · Markdown table 2.29 · Chinese 1.44, a **4.5× spread**. chars ÷ 4 over-estimates prose and under-estimates tables (~1.7×) and Chinese (~2.8×). Numbers are tokenized digit by digit (`128K` → 4 tokens). `<|im_start|>` is a single special token (id 151644).
- **Decision:** Two lines of defence:
  1. **Prevention.** The exact Qwen2.5 tokenizer counts every piece of the prompt, and the prompt builder never exceeds the budget.
  2. **Detection.** After each response, compare Ollama's `prompt_eval_count` with our expected count. If Ollama processed noticeably fewer tokens, it truncated: log a loud warning. If the two persistently disagree by more than the per-message template estimate, the tokenizer file doesn't match the running model (drift).
- **Loading (agreed 2026-11-21, M1):** `tokens.py` loads the tokenizer once with `Tokenizer.from_file(config.TOKENIZER_PATH)`, from a local `tokenizer.json` that a one-time setup step downloads at a **pinned revision** into a gitignored `models/` folder. We rejected `Tokenizer.from_pretrained(...)` (used in the experiments): it can contact the Hugging Face Hub every time it runs unless `HF_HUB_OFFLINE` is set, which weakens the "fully local at runtime" claim and could stop chat from starting offline. We also rejected committing the file to git, because it's a generated artifact. Besides `count(text)`, `tokens.py` provides the message-level formula `Σ count(content) + 5·n_messages + 3`. It's the only module that knows which tokenizer is in use.
- **Tokenizer ↔ model coupling:** the config sets the chat model and its tokenizer side by side. Changing the model without changing the tokenizer is a bug the detection check should catch.
- **Caching caveat: resolved (measured).** In Ollama 0.40.16, `prompt_eval_count` always reports the **full** prompt, even when most of it is served from the prefix cache. The cached part is reported separately as `prompt_eval_cached_count`. So the detection check is reliable on every request.
- **Why:** Exact counts make the 6K guarantee provable rather than approximate, and the detection check turns Ollama's silent failure mode into a visible one.
- **Subtlety:** Ollama wraps each message in Qwen's ChatML template (`<|im_start|>role\n … <|im_end|>\n`), which adds overhead. **Measured: exactly 5 tokens per message + 3** for the trailing assistant header, so `prompt_tokens = Σ count(content) + 5·n_messages + 3`. That matched `prompt_eval_count` exactly. Without a system message, Qwen's template injects a default one (+21 tokens), so we always send our own.
- **Likely reviewer questions:**
  - *"Is downloading a tokenizer still 'fully local'?"* Yes. It's a static file fetched once at setup, the same as the model weights. Nothing goes over the network at runtime.
  - *"How do you know your count matches Ollama's?"* We compare our count with `prompt_eval_count` on real requests, and a test fails if they drift beyond the margin. Measured in M0: once the `5·n_messages + 3` template overhead is added, they match **exactly**.

### D-011: Budget allocation: fixed caps
- **Status:** accepted
- **Options considered:**
  - **Fixed caps** (see the table at the top). Pros: predictable, easy to explain and unit-test. Cons: any unused history allowance is wasted rather than given to context.
  - **Dynamic, context first.** Pros: uses every token. Cons: history can vanish entirely on queries with many relevant chunks, and behaviour is harder to predict.
  - **Dynamic, history first.** Pros: good for long chats. Cons: starves retrieval, and documents are the point of a RAG app.
- **Decision:** Fixed caps: answer 1,024 · system ~300 · history ≤ 1,024 · margin ~50 · context ≤ 3,746. (Revised 2026-11-21: the margin went from ~200 to ~50 once M0 showed our counts are exact. We considered lowering the context ceiling to ~2,500 for faster answers and kept it; the relevance filter (D-015) is what keeps typical prompts short.)
- **Prompt order (added 2026-11-21): `system → history → context + question`.** Ollama reuses its KV cache for the longest identical **prefix** of consecutive prompts, and recomputes everything after the first changed token (measured: a one-word change at 75% depth → 75% cached and prefill 10.8 s → 2.8 s; at 25% depth → 26% cached). System and history are identical from one turn to the next, so putting them first makes them a cached prefix. Only the new context and question pay prefill. Putting context before history would break the prefix every turn and force history to be recomputed.
- **Why:** Each slot has a clear owner and a hard limit. That makes "how do you stay under 6K?" a one-line answer, and the property easy to test.
- **Where the caps and the system prompt live (agreed 2026-11-21, M1):** every cap is a constant in `config.py` (constants only, no logic). That includes `SYSTEM_PROMPT_CAP = 300`, so the whole budget can be read in one place. The system prompt **text** lives in `prompt.py` (M3), next to the code that assembles the prompt. We rejected putting it in `config.py` (a long prose string in a constants file) and in a separate `prompts/system.txt` (an extra file and path for a single prompt).
- **The question's tokens (amended 2026-11-23, M3):** the original caps gave the user's question no slot, so a long question could eat the 50-token margin and overflow the window. Now the question is charged to the context slot: chunks get `budget − count(question + wrapper)`. The invariant `prompt.py` enforces and tests is **total prompt ≤ 5,120 tokens** (`NUM_CTX − num_predict`), counted with the exact message formula (D-010). If the question alone doesn't fit, that's an error, not a silent truncation.
- **Principle: the budget is a ceiling, not a target** (added 2026-11-20). Unused tokens are not handed to another slot. A shorter prompt means faster prefill, which is the real benefit of filtering out irrelevant chunks (D-015). We considered letting history borrow tokens that context leaves unused, and rejected it: it would mean more stale conversation *and* slower answers.
- **Trade-offs accepted:** We leave tokens unused when history or context is short. That's deliberate (see the principle above).
- **Likely reviewer questions:**
  - *"Why not use every available token?"* Prefill time grows with prompt length (measured on the M2 Pro: prefill ≈ 280 tok/s, so a ~5K-token prompt takes **~18.6 s** before the first token). Tokens with no relevant information cost latency and can distract the model.
  - *"Why reserve 1,024 tokens for the answer?"* Generated tokens occupy the same context window as the prompt. **Measured:** a 6,022-token prompt with `num_predict=300` still generated 300 tokens. Ollama silently evicted the *start* of the prompt mid-answer (context shifting) to make room, which is where the system prompt sits. Keeping prompt ≤ 6,144 − `num_predict` guarantees that never happens. 1,024 tokens is roughly 700 words, plenty for a grounded answer.
  - *"Why does context get the largest share?"* Answer quality in RAG depends mostly on whether the right passage is in the prompt.

---

## Retrieval

### D-012: Ingestion: a separate `ingest` command with an on-disk index
- **Status:** accepted
- **Options considered:**
  - **`ingest` command writing `index/vectors.npy` + `index/chunks.json`.** Pros: separates indexing from querying; chat starts instantly; the index can be inspected. Cons: the index can go stale if files change and nobody re-runs ingest.
  - **Rebuild on every start.** Pros: never stale; no files. Cons: re-embeds everything on every launch, so the demo starts with a wait.
  - **Incremental (content hashes).** Pros: fast re-ingest. Cons: change tracking and deletion handling for a corpus that rebuilds in seconds.
- **Decision:** A full rebuild on `ingest`, saved to `index/`. The metadata records the embedding model, chunking parameters and creation time.
- **Why:** It's the standard offline-index/online-query split, and a small corpus makes full rebuilds cheap.
- **Trade-offs accepted:** Staleness. Mitigation: `chat` warns if any source file is newer than the index.
- **Staleness check (built in M4, 2026-11-23):** `ingest` records the absolute `data_dir` in `manifest.json`. `store.load` warns if any `.md`/`.txt` under it has a newer mtime than `manifest.json`. It only warns and still loads (like the chunk-param check), because a stale index still gives correct answers for the files that didn't change. Limit: a *deleted* file doesn't trigger it, since there's no newer mtime to see.
- **Load checks (M2):**
  - `manifest.json` records `embed_model`, `chunk_size`, `overlap` and `created_at`.
  - **Refuse** to load if `embed_model` differs from config: vectors from a different model live in a different space, so search would be meaningless.
  - **Warn only** if the chunk parameters differ: the index is stale but still correct.
  - **Raise** if `len(vectors) != len(chunks)`: row *i* must be chunk *i*, and a half-written or hand-edited index would otherwise give wrong source labels with no error.
  - Files are written only after every embedding has succeeded.
- **Likely reviewer questions:**
  - *"What happens if I edit a file?"* Re-run `ingest`, and chat warns you if you forget. Incremental indexing by content hash is the upgrade path.

### D-013: Retrieval query: weighted blend of current and previous question
- **Status:** accepted (revised 2026-11-20: was "previous + current question as joined text")
- **Context:** A follow-up like "what about its limitations?" has no subject, so embedding it alone retrieves noise. But adding the previous question can drag retrieval back to the *old* topic when the user changes subject.
- **Options considered:**
  - **Joined text** (embed `previous + "\n" + current`). Pros: one embedding call; resolves pronoun follow-ups. Cons: the two questions carry equal weight, so **topic switches retrieve the old topic**. Toy example: after "How do I turn on sync?", the question "When does it launch?" ranked the *Enable sync* chunk (0.62) above the launch-date chunk (0.52).
  - **Weighted vectors.** Embed each question separately, then `q = normalize(w·q_curr + (1−w)·q_prev)`. Pros: the current question dominates when it has its own topic, and the previous one fills in a missing subject. In the same toy example at w = 0.8, both the topic switch (launch-date chunk first, 0.67) and the pronoun follow-up (*Limits* chunk first, 0.85) work. Cons: a second embedding call (~20 ms) and one parameter to tune.
  - **Raw current question.** Pros: simplest. Cons: fails on follow-ups, which reviewers will try.
  - **LLM rewrites the question to stand alone.** Pros: most accurate, and handles chains of vague follow-ups. Cons: an extra LLM call (~1–3 s), another prompt to budget, and the rewrite can itself go wrong.
- **Decision:** Weighted vectors: `q = normalize(w·embed(current) + (1−w)·embed(previous))`, starting at **w = 0.8**, tuned on the eval set (D-019). On the first turn, `q = embed(current)`.
- **Why:** It fixes the topic-switch failure of joined text at almost no cost, and makes the trade-off between "current" and "context" one explicit, tunable number.
- **Measured (M4, `eval/run_eval.py`, 5 follow-up turns):** hit@1 is 0.40 / 0.60 / 0.60 / **1.00** / 0.60 / 0.60 at w = 0.5 / 0.6 / 0.7 / **0.8** / 0.9 / 1.0. w = 1.0 (current question only) misses the pronoun follow-ups; w ≤ 0.6 lets the old topic win after a switch. **Kept w = 0.8.** It's a small sample, so the claim is "0.8 is in a good range", not "0.8 is optimal".
- **Trade-offs accepted:** If the *previous* question was itself vague, its vector adds noise rather than a subject. LLM rewriting is the upgrade path if the eval shows this matters.
- **Likely reviewer questions:**
  - *"Why not use the whole history?"* Older turns increasingly blur the query. The previous question alone carries most of the reference.
  - *"Why is averaging vectors meaningful?"* Embeddings live in a space where direction encodes meaning. A weighted sum points *between* the two topics, closer to the heavier one. We renormalize so cosine scores stay comparable.
  - *"How did you choose 0.8?"* Starting value, then tuned on eval questions that include both follow-ups and topic switches.

### D-014: Chunking: structure-aware, ~400 tokens, ~50 overlap, heading-path prefix
- **Status:** accepted
- **Options considered:**
  - **Structure-aware.** Split on Markdown headings, then paragraphs, then sentences, packing pieces up to ~400 tokens. Overlap of ~50 tokens applies only when a section is cut into several chunks. Pros: chunks are coherent units. The heading prefix (`Guide > Install > macOS`) restores context a chunk would otherwise lose. Code blocks and tables stay whole where possible. Cons: more code; edge cases (headings inside code fences, very long paragraphs).
  - **Fixed token windows.** Pros: simplest. Cons: cuts mid-sentence, mid-code-block and mid-table, and chunks lose their section context.
  - **Smaller (~200 tokens).** Pros: more precise matches, and about 15 fit in the budget. Cons: each chunk has less surrounding context for the LLM.
- **Decision:** Structure-aware, ~400 tokens with ~50 overlap. `.txt` files have no headings, so they fall back to paragraph/sentence splitting with the filename as the prefix.
- **Why:** About 400 tokens is big enough to hold a full explanation and small enough that 6–8 chunks fit in about 3.6K, which gives retrieval several chances to hit. There's also a deeper reason: **pooling dilutes meaning.** The embedding model runs attention over every token (a tokens × 768 matrix), then *averages* the rows into one vector. A chunk covering install + sync + pricing becomes a blurry average that matches none of those questions strongly. (Toy example: a mixed chunk scored ~0.62 against "How do I turn on sync?", while a focused sync chunk scored 0.97.) Smaller, single-topic chunks give sharper vectors.
- **Measured (M4):** hit@1 / hit@3 / hit@5 on 18 answerable turns: 200 tokens 0.78 / 0.94 / 1.00 · **400 tokens 0.94 / 1.00 / 1.00** · 600 tokens 0.89 / 1.00 / 1.00. 200-token chunks split explanations, so the right section ranks lower. 600-token chunks mix topics (the pooling effect above). **Kept 400.** On the demo corpus: 245 chunks, mean 180 tokens, max 400.
- **Implementation details (agreed 2026-11-21, M1):**
  - **Data shape.** `chunk(Document) -> list[Chunk]` is a pure function: no disk access, no embeddings. `Chunk(source, heading_path: list[str], text, n_tokens)`. Each field has a reader: `source` is used by the sources list (D-017); `heading_path` by the embedded prefix and the sources list; `text` by the embedder and the prompt; `n_tokens` by budget packing (D-015). The string that gets embedded and put in the prompt is *derived*: `" > ".join(heading_path) + "\n" + text`. It isn't stored, so prefix and body can't drift apart.
  - **`n_tokens` counts prefix + body** (Qwen tokens), because that's what enters the prompt. Counting only the body would under-count by about 12 tokens per chunk; across 8 chunks that's bigger than the 50-token margin, and the result would be silent context shifting (D-011). `n_tokens` is a *packing estimate*. `prompt.py` still counts the fully assembled messages for the final guarantee (token counts aren't exactly additive across concatenation, and separators cost tokens).
  - **Heading detection.** A line is a heading if it starts with 1–6 `#` followed by a space. A heading stack gives the path: a level-N heading pops the stack down to N−1 entries, then pushes itself.
  - **Code fences are atomic.** An `in_fence` flag (toggled by lines starting with `` ``` `` or `~~~`) turns off **both** heading detection and paragraph splitting inside fences. Otherwise `# install deps` in a bash block becomes a fake heading, and a blank line between two functions breaks the block into two broken fences. The section is scanned once into **blocks** (paragraphs and fenced code).
  - **Split order:** heading → block → sentence → **hard Qwen-token split as a last resort** for a single block still over ~400 tokens (e.g. a large table or code block). This guarantees every chunk is at most about 400 tokens + prefix. An oversized **code** block skips the sentence step and goes straight to the token split (code has no prose sentences).
  - **Known limit, deferred (2026-11-22):** a token split can cut mid-line. In a table that splits a row (`| E1014 | Sync request 14` / `failed after retrying |`); in code it splits a command and leaves a fence unbalanced, which could make the model read later prompt text as code. We considered a **line-level step** (prose: sentence → line → token; code: line → token) and **fence repair / header-row repeat**, and deferred both as extra complexity: no block in the current `data/` exceeds the budget. **Re-checked in M4 (2026-11-23):** the corpus has 161 code blocks and tables, and none exceeds 350 tokens, so the token split never cuts a code block or table. The line step stays deferred.
  - **Pieces carry facts, not formatting:** `Piece(text, starts_block, code)`. `_join` decides that a new block is glued with `"\n\n"` and a continuation with `""` (sentence and token splits keep the original whitespace, so re-joining reproduces the text exactly).
  - **Sentences:** a regex split after `.`, `!` or `?` followed by whitespace. Abbreviations (`e.g.`) cause early cuts. That's harmless, because the pieces are re-packed up to ~400 tokens.
  - **Overlap:** the **whole trailing sentences** of the previous chunk that fit in ≤50 tokens, so the overlap stays readable. If the last sentence alone is over 50 tokens, that cut gets no overlap.
  - **No heading above the text** (`.md` preamble, `.md` with no headings, every `.txt`): `heading_path = [filename]`. Otherwise the path holds headings only, with no filename.
  - Options rejected: storing `full_text` as a field (duplicated data); keeping oversized blocks whole (a 4,000-token table exceeds the 3,746 context cap and could **never** be retrieved); exact last-50-token overlap (chunks start mid-sentence); `pysbd`/`nltk` (a dependency to protect cut points that re-packing makes almost irrelevant); always prepending the filename (costs tokens and usually repeats the H1).
- **Likely reviewer questions:**
  - *"Why overlap?"* So a fact that straddles a cut point appears whole in at least one chunk.
  - *"Why prefix the heading path?"* A chunk under `## Limitations` doesn't say *what* it's describing the limitations of. The prefix gets embedded with the text, so the chunk can be retrieved by topic.
  - *"Isn't the hard token split exactly the fixed-window approach you rejected?"* No. Fixed windows are rejected **as the default**. Here almost all text is cut at headings, blocks and sentences, and only a single block too big to split any other way reaches the token split. Kept whole, it couldn't fit the context cap and would be unreachable. *"A half-table you can retrieve beats a whole table you can't."* The next step would be splitting on line boundaries (whole table rows, whole code lines) before falling back to tokens, then repeating the header row. Deferred because no block in our corpus needs it yet (see the known limit above).
  - *"Your chunks are 400 tokens by the wrong tokenizer. nomic embeds them."* Chunk size exists to fit the **LLM's** prompt, so it's measured in Qwen tokens. nomic only needs the chunk to fit its input window, which is far larger. **[verify] in M2:** nomic's effective window in Ollama, and that `/api/embed`'s `prompt_eval_count` shows no chunk is truncated.

### D-015: Chunk selection: best-first, filtered by relevance, capped by budget
- **Status:** accepted (revised 2026-11-20: was "fill until the budget is full")
- **Context:** In the walkthrough example, "fill until full" let an irrelevant *Install* chunk (score 0.30) into a question about sync, just because there was room. That adds noise and makes prefill slower.
- **Options considered:**
  - **Fill by score until the context cap (~3,746 tokens).** Pros: uses the budget fully. Cons: always sends the maximum prompt (the slowest prefill) and lets in irrelevant chunks.
  - **Absolute score floor** (e.g. keep score ≥ 0.5). Pros: detects "nothing relevant at all". Cons: cosine scores have no absolute scale; the right number depends on the embedding model and corpus.
  - **Relative cutoff** (keep score ≥ top − δ). Pros: adapts to each query's score range. Cons: when nothing is relevant it still keeps the "best of the bad".
  - **Fixed k = 4.** Pros: simplest to explain. Cons: ignores both relevance and chunk length.
  - **Over-fetch 20 + cross-encoder rerank.** Pros: best precision. Cons: a second model plus PyTorch, overkill for a small corpus.
- **Decision:** Rank by cosine score. Add chunks best-first **while score ≥ floor AND score ≥ top − δ**, stopping at the context budget (~3,746 tokens) or 8 chunks, whichever comes first. A chunk that doesn't fit the remaining budget is skipped and the next one tried. Floor and δ start as guesses and are **tuned on the eval set** (D-019). Example (scores 0.97 / 0.82 / 0.41 / 0.30, δ = 0.2): only the first two pass.
- **Who does what (amended 2026-11-23, M3):** the check is split in two. `retriever.select()` applies **relevance only**: floor, δ, max 8 chunks. It returns `[(chunk, score)]` best-first and never counts tokens. `prompt.py` then fits those chunks into the token budget best-first, skipping any chunk that doesn't fit and trying the next. It returns the chunks it actually included, and those are what the source list shows (D-017).
  - *Options:* (A) the retriever also checks the budget, or (B) budget fitting lives in `prompt.py`. **Chose B.** `prompt.py` adds a label and separators around each chunk, so only it knows a chunk's real cost. Example: a 205-token chunk plus a 20-token label costs 225. With 220 tokens left, A counts 205, keeps the chunk and overflows. B counts 225 and skips it. B also keeps every token count in one module, so the test "the prompt never exceeds the budget" covers every path.
  - *Skip vs stop:* when chunk 3 doesn't fit but chunk 4 does, chunk 4 goes in. It already passed the relevance filters, and stopping would leave budget unused. Rare in practice (8 × ~400 ≈ 3,200 < 3,746). It only happens with full history, a long question or an oversized chunk, so it gets its own test.
- **Why:** The budget is a ceiling, not a target (D-011). Relevance decides what gets in, and the budget only guarantees we never overflow. The floor catches "nothing relevant", and the relative cutoff handles per-query score scales.
- **Measured (M4):** top scores of answerable turns range from 0.654 to 0.859. Off-topic questions top out at 0.555 (capital of Australia), 0.643 (banana bread) and **0.669** (Kubernetes, because "configure" matches MkDocs' configuration pages). So **no floor separates them cleanly**. A floor of 0.65 would keep recall at 1.00 but sits only 0.004 below the weakest answerable question, which is overfitting to 22 turns. **Chose floor 0.60** (a 0.05 margin; it filters the weakest off-topic question) and **δ 0.10**: recall stays 1.00, with 5.7 chunks sent on average instead of 7.2 at δ 0.20. Since cosine scores can't separate everything, the off-topic questions that pass the floor are handled by D-016's prompt grounding. Live test: "How do I configure Kubernetes autoscaling?" gets 8 MkDocs chunks (0.64–0.67) and the model replies that the documents don't cover it. An on-domain unanswerable question ("Does Lumen have an Android app?", top 0.80) is the same case.
- **Trade-offs accepted:** Two parameters to tune, and a cutoff set too strict can drop a useful chunk. If no chunk passes the floor, the prompt has no context, and the model must say it doesn't know (D-016).
- **Likely reviewer questions:**
  - *"What about 'lost in the middle'?"* LLMs pay most attention to the start and end of the context. With only 6–8 chunks the effect is small, and putting the best chunk closest to the question is a cheap fix. (Chunk order within the prompt is a detail we'll settle while building.)

### D-016: No relevant match: grounding by prompt only
- **Status:** accepted
- **Options considered:**
  - **Prompt only.** Always pass the best chunks, and instruct the model to answer only from the context and otherwise say it doesn't know. Pros: no threshold to tune. Cons: a 7B model sometimes answers from its own knowledge anyway.
  - **Similarity threshold → refuse.** Pros: deterministic. Cons: cosine thresholds are specific to the model and corpus, and set too high, they refuse good questions.
  - **Threshold → answer but flag it.** Pros: friendly. Cons: blurs the RAG guarantee.
- **Decision:** Grounding by prompt, **plus the relevance floor from D-015** (amended 2026-11-20). If no chunk passes the floor, we send no context, and the system prompt tells the model to say it doesn't know from the documents.
- **Why:** The prompt handles the common case. The floor, calibrated on the eval set, removes irrelevant chunks the model might otherwise wrongly build an answer from. The fictional `.txt` docs (D-022) give us a direct test of whether the model sticks to context.
- **Trade-offs accepted:** A 7B model may still occasionally answer from its own knowledge, even with empty context. If testing shows that's frequent, the CLI could skip the LLM and reply "nothing relevant found" when no chunk passes.
- **Likely reviewer questions:**
  - *"How do you stop hallucinations?"* We reduce them: a strict system prompt, sources shown under each answer, and fictional docs to test grounding. Prompting can't fully prevent them, and a score threshold is the next layer.

### D-017: Sources: a list printed after each answer
- **Status:** accepted
- **Options considered:**
  - **Source list from metadata.** Pros: zero tokens; can't be hallucinated. Cons: shows what was *retrieved*, not necessarily what was *used*.
  - **Inline `[n]` citations.** Pros: links each claim to its source. Cons: 7B models often misnumber or skip them; costs instruction tokens.
  - **None.** Pros: clean output. Cons: grounding can't be verified, and it removes the easiest demo of RAG working.
- **Decision:** Print `file › heading path (score)` for each chunk in the prompt, below the streamed answer.
- **Likely reviewer questions:**
  - *"These are retrieved sources, not cited ones?"* Correct, and we're upfront about that. Inline citations are the upgrade, at the cost of reliability with 7B models.

---

## Engineering

### D-018: Code layout: a package with one module per stage
- **Status:** accepted
- **Options considered:**
  - **`rag_chat/` package** (`config`, `loader`, `chunker`, `tokens`, `embedder`, `store`, `retriever`, `prompt`, `llm`, `cli`). Pros: each stage testable in isolation; the review walkthrough follows the data flow. Cons: more files.
  - **Single script.** Pros: fastest start. Cons: hard to test and hard to walk through.
  - **Ports/adapters, with interfaces for the LLM, embedder and store.** Pros: shows the system can swap components. Cons: abstractions with a single implementation are code to defend without benefit.
- **Decision:** The package, using plain functions and small classes, with no abstract interfaces until a second implementation exists.
- **Likely reviewer questions:**
  - *"How would you swap Ollama for llama.cpp?"* Only `llm.py`, `embedder.py` and (for counting) `tokens.py` know about Ollama, so the change stays inside those modules.

### D-019: Testing: pytest unit tests + retrieval eval
- **Status:** accepted
- **Options considered:**
  - **Unit tests + retrieval eval.** Pros: proves the deterministic properties hold, and gives evidence for tuning choices. Cons: writing about 15 eval questions takes time.
  - **Unit tests only.** Pros: less work. Cons: chunk size and k stay guesses.
  - **Also an LLM-judged answer eval.** Pros: measures answer faithfulness. Cons: slow and noisy with a 7B judge; a big extra build.
- **Decision:** pytest for the chunker, tokenizer budget (**the assembled prompt never exceeds 6,144 − 1,024**), history window and store. Plus `eval/questions.jsonl` with about 15 questions, each labelled with its expected source file, scored as **hit-rate@k**. The eval set must include **follow-up questions and topic switches** (short conversations, not just single questions), plus a few **unanswerable questions**. Together these tune the query weight `w` (D-013) and the relevance floor and `δ` (D-015).
- **Why:** Gives evidence for the claim that we stay under 6K, and turns the guessed numbers (chunk size, `w`, floor, `δ`) into measured ones.
- **Likely reviewer questions:**
  - *"How did you pick 400 tokens?"* We ran the eval at 200, 400 and 600: hit@1 was 0.78, 0.94 and 0.89 (D-014).
- **Built (M4):** `eval/questions.jsonl` has 15 conversations (22 turns: 18 answerable, 4 unanswerable). An expected source is `file` or `file#heading-substring`, because one Lumen file holds several topics, so the eval checks the right *section*. `python -m eval.run_eval` runs the real loader → chunker → embedder → retriever in memory (`index/` is untouched) and sweeps chunk size, w, floor and δ. Limits: 22 turns is small, and it measures retrieval only, not answer quality.

### D-020: Streaming output
- **Status:** accepted
- **Decision:** Stream tokens from Ollama (`stream: true`) and print them as they arrive. Sources print after the stream ends.
- **Why:** A 300–1,000-token answer at ~25 tok/s takes 12–40 s. Without streaming the demo shows a frozen terminal. Streaming also improves time to the first token, which is how users perceive latency.
- **Trade-offs accepted:** Slightly more complex code: handling partial chunks, Ctrl-C mid-stream, and errors mid-stream.
- **Built (M4):** Ctrl-C mid-answer prints `[interrupted]` and returns to the prompt, and the half answer stays out of the history. `chat` also warms up the model at start (an empty `/api/chat` request with the same `num_ctx`, so Ollama doesn't reload the model on the first question).

### D-021: Setup: venv + pinned `requirements.txt`
- **Status:** accepted
- **Options considered:**
  - **venv + pinned requirements.** Pros: standard, needs no extra tools. Cons: no lockfile for transitive dependencies (fine with so few dependencies).
  - **pyproject + uv.** Pros: fast and reproducible. Cons: reviewers need uv installed.
  - **Docker.** Pros: reproducible. Cons: **Docker on macOS can't use the Metal GPU**, so Ollama in a container would run on CPU only and be far too slow.
- **Decision:** venv + pinned `requirements.txt`. Setup steps: install Ollama → `ollama pull qwen2.5:7b-instruct nomic-embed-text` → venv + `pip install` → a one-time tokenizer download (a pinned `tokenizer.json` saved to the gitignored `models/` folder, see D-010) → `ingest` → `chat`.

### D-022: Demo corpus: public Markdown docs + fictional `.txt` notes
- **Status:** accepted (revised: originally public docs only)
- **Context:** The demo must show both `.md` **and** `.txt` ingestion, and must prove that answers come from retrieval rather than from the model's training data.
- **Options considered:**
  - **Public docs only.** Pros: realistic Markdown structure. Cons: no `.txt` files; Qwen may already know these docs, so a correct answer doesn't prove retrieval worked.
  - **Fictional only.** Pros: the model can't know the answers. Cons: writing 20+ realistic files takes a lot of time.
  - **Mixed.** Pros: covers both formats plus a grounding proof. Cons: a little more work.
- **Decision:** A subset of permissively licensed open-source Markdown docs (tens of files with headings, code blocks and tables), **plus a few fictional `.txt` documents**, such as internal notes for a made-up company or product, containing specific facts the model cannot have seen.
- **Why:** The fictional docs give a decisive demo moment: ask a question only they can answer, get the right answer, and see the `.txt` source listed. Asking something in neither set shows the model saying it doesn't know.
- **Chosen (M4):** 13 pages of the **MkDocs 1.6.1 docs** (BSD-2-Clause, licence file kept in `data/mkdocs/LICENSE`, attributed in the README). Why MkDocs: plain Markdown with deep heading trees, many code blocks and option tables, and a permissive licence. Left out: pages that are only build directives (`::: mkdocs-click`, `--8<--` includes) and the 110 KB changelog, which would flood retrieval with version-note noise. Fictional files: `lumen_guide.md`, `support_playbook.txt`, `team_notes.txt`.
- **Likely reviewer questions:**
  - *"How do I know the model isn't just answering from pretraining?"* Ask about the fictional company. Those facts exist only in our `.txt` files.

### D-023: Loader policy for unreadable and empty files
- **Status:** accepted (2026-11-21, M1)
- **Context:** `loader.py` finds `.md`/`.txt` files under a directory and returns `Document(path, text)`. It does **no retrieval** (that happens in `retriever.py`, against the index). A real `data/` folder can contain a Latin-1 file, a file with a BOM, or an empty placeholder.
- **Options considered:**
  - **Skip and warn.** Read as `utf-8-sig` (which strips a BOM). On `UnicodeDecodeError`, or text that is empty or only whitespace, skip the file, print a warning naming it, and continue. Pros: one bad file never blocks ingest, and nothing garbled reaches the index. Cons: a skipped file's content is missing until it's fixed (the warning makes that visible).
  - **`errors="replace"`.** Pros: never fails. Cons: bad bytes become `U+FFFD`, and the garbled text gets embedded, retrieved and shown to the LLM, all silently.
  - **Fail the whole ingest.** Pros: strictest. Cons: one stray file stops everything, which is risky in a live demo.
- **Decision:** Skip and warn.
- **Why:** It matches the project's rule that **no failure is silent** (as with Ollama truncation in D-010 and D-011), without letting one file stop the whole run.
- **Likely reviewer questions:**
  - *"Why not guess the encoding?"* Guessing (e.g. with `chardet`) can be confidently wrong and adds a dependency. A clear warning lets the user convert the file once.
