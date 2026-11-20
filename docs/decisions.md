# Design Decisions

A log of the architectural decisions behind this project, kept as prep for the code review.

**Status legend**
- `accepted — deep-dive pending`: agreed in a quick design round. Ryan has not yet challenged it in depth, and it may still change.
- `accepted`: discussed in depth. Ryan can defend it.
- `superseded by D-XXX`: replaced by a later decision.

Facts marked **[verify]** come from documentation or memory and have not yet been checked against the real system. Confirm each one when we build that component.

---

## Architecture at a glance

```
                     INGEST (offline, `rag-chat ingest <dir>`)
 .md/.txt files ──► loaders ──► chunker ──► embedder ──────► index/ (vectors.npy + chunks.json)
                                (heading-aware,  (Ollama:
                                 ~400 tok)        nomic-embed-text)

                     CHAT (interactive, `rag-chat chat`)
 user question ─┬─► query = prev question + current ─► embedder ─► NumPy cosine search
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
| Safety margin | ~200 | Chat-template tokens and counting error. |
| **Retrieved context** | **≈ 3,596** | Whatever is left. Filled best-first (D-015). |
| **Total** | **6,144** | Sent as `num_ctx` on every request. |

---

## Foundations

### D-001: Language: Python
- **Status:** accepted — deep-dive pending
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
- **Status:** accepted — deep-dive pending
- **Context:** The LLM must run locally on an Apple M2 Pro with 16 GB of unified memory. Most local runtimes run quantized **GGUF** models through **llama.cpp**, which uses the Apple GPU via Metal. The real question is how our app talks to llama.cpp.
- **Options considered:**
  - **Ollama.** A local server wrapping llama.cpp, with a REST API on `localhost:11434`.
    - Pros: one-command install. `ollama pull <model>` handles downloads, quantization variants and chat templates. Metal acceleration is automatic. The model runs as a **separate process**, so the app is a thin HTTP client and models swap with one config value. Easy for reviewers to reproduce. It also serves embeddings (D-007), so one runtime covers both.
    - Cons: a background service is one more moving part. **No public tokenize endpoint**, so we count tokens ourselves (D-010). Its **default context window is smaller than 6K** (2,048 or 4,096 depending on version **[verify]**), so we must send `num_ctx: 6144` with every request. When a prompt overflows, Ollama **truncates it silently** (only a server-side log warning) and drops early prompt content, which is where our system prompt sits **[verify behaviour empirically]**. We also give up some low-level control (sampling internals, KV-cache reuse).
  - **llama-cpp-python.** Python bindings that run llama.cpp **inside our process**.
    - Pros: `Llama(n_ctx=6144)` sets the window directly. **`llm.tokenize()` gives exact counts with the exact model tokenizer**, with no extra dependency. No separate service to run, and full control over sampling.
    - Cons: the pip install compiles C++ with Metal flags, which can fail on some machines (a demo risk; prebuilt Metal wheels exist but their availability varies **[verify]**). We download GGUF files by hand from Hugging Face and pick the quantization ourselves. Our app manages the model's lifecycle and memory, and the model loads on every app start (several seconds). Chat templates are more manual.
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
- **Why:** The lowest-risk path to a working demo for someone new to local LLMs. It separates model serving from application logic cleanly, and reviewers can reproduce it with `ollama pull`. Its two weaknesses (context size and token counting) are things we take on explicitly, and they make good review talking points.
- **Trade-offs accepted:** We own 6K enforcement: `num_ctx` on every call plus our own exact token counting, with a test that the assembled prompt is always ≤ 6,144 tokens. We give up the in-process `tokenize()` that llama-cpp-python would provide.
- **Likely reviewer questions:**
  - *"The model supports 32K natively. How do you guarantee 6K?"* Two layers. (1) The runtime layer: `num_ctx=6144` on every request means the KV cache is physically 6K. (2) The application layer: the prompt builder counts tokens exactly and never assembles more than 6,144 − 1,024 (the answer reserve) tokens. A unit test asserts this.
  - *"What happens if you overflow?"* Ollama silently truncates, the system prompt gets dropped, and the model stops following grounding rules. That's why we enforce the limit *before* sending rather than relying on the runtime.
  - *"Why not llama.cpp directly?"* We'd gain exact tokenization in-process, but take on the compile risk and model-file management. Ollama is llama.cpp underneath, so performance is the same.

### D-003: No RAG framework; we write the pipeline ourselves
- **Status:** accepted — deep-dive pending
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
- **Status:** accepted — deep-dive pending
- **Context:** Corpus size drives the choice of vector store, ingestion design and performance work.
- **Options considered:** small demo corpus (tens of files, a few MB); reviewer-supplied files at demo time; large (thousands of files).
- **Decision:** Small: tens of files that we curate (see D-022).
- **Why:** That's what the demo needs. Designing for thousands of files would add an approximate-nearest-neighbour (ANN) index, incremental indexing and batching that we'd have to build and defend without needing them.
- **Trade-offs accepted:** Not tuned for large corpora, but the upgrade path is clear and explainable (D-008).
- **Likely reviewer questions:**
  - *"What if I drop in 10,000 files?"* Ingestion slows down linearly because embedding is the bottleneck, and brute-force search still works up to about 100K chunks. Beyond that we'd move to FAISS/HNSW and incremental indexing.

### D-005: Interface: CLI chat
- **Status:** accepted — deep-dive pending
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
- **Status:** accepted — deep-dive pending
- **Context:** The model must fit in 16 GB alongside the OS and the embedding model, run fast enough for a live demo, and stick to the provided context.
- **Memory math:** 7B parameters at ~4.5 bits each ≈ 4.7 GB of weights. KV cache at 6K tokens for Qwen2.5-7B (28 layers, 4 KV heads × 128 dims, fp16) ≈ 2 × 28 × 4 × 128 × 2 bytes ≈ 56 KB per token × 6,144 ≈ **~340 MB**. Total ≈ 5–5.5 GB, which fits comfortably.
- **Options considered:**
  - **qwen2.5:7b-instruct.** Pros: strong instruction-following and context use; Apache-2.0 licence; **no "thinking" mode**, so no hidden reasoning tokens eat the answer budget; its tokenizer is downloadable without a licence gate. Cons: not the newest model; it was trained with a default system prompt in its template, so we must always send our own.
  - **llama3.1:8b-instruct.** Pros: very well known. Cons: slightly larger; its Hugging Face tokenizer sits behind a licence click, which makes exact counting awkward (D-010).
  - **qwen3:8b.** Pros: newer and stronger at reasoning. Cons: "thinking" mode is on by default and emits long reasoning before answering, which consumes the 1,024-token answer reserve unless disabled.
  - **gemma3:4b.** Pros: about 3 GB and faster. Cons: weaker at strict grounding and at saying "I don't know".
  - Larger (12–14B). Pros: better quality. Cons: about 8–9 GB of weights on a 16 GB machine; slower generation and memory pressure in a live demo.
- **Decision:** `qwen2.5:7b-instruct`, configurable.
- **Why:** The best balance of grounding quality, speed (roughly 20–30 tok/s expected on an M2 Pro **[verify]**) and a predictable token budget.
- **Trade-offs accepted:** The tokenizer is tied to the model family. Switching to a non-Qwen model means switching the tokenizer file too (D-010).
- **Likely reviewer questions:**
  - *"Why 4-bit quantization? Doesn't it hurt quality?"* Q4_K_M is the usual sweet spot. Quality loss compared with fp16 is small for 7B models, and it cuts memory by about 3.5×.
  - *"Why not a bigger model?"* Memory and latency on a 16 GB laptop. A 7B model with good retrieval beats a 14B model that stutters in a demo.

### D-007: Embeddings: `nomic-embed-text` via Ollama
- **Status:** accepted — deep-dive pending
- **Context:** Retrieval needs a local model that turns text into vectors whose cosine similarity reflects meaning.
- **Options considered:**
  - **nomic-embed-text (Ollama).** Pros: reuses the runtime we already run; no PyTorch (~2 GB) install; 768-dim vectors; long input limit (trained to 8K tokens; the limit Ollama actually applies is **[verify]**). Cons: needs task prefixes (`search_document: ` when indexing, `search_query: ` when searching), and forgetting them quietly degrades retrieval. An HTTP call per batch.
  - **sentence-transformers (bge-small, all-MiniLM-L6-v2).** Pros: in-process, hugely popular, many model choices. Cons: PyTorch dependency; a second model-loading path; MiniLM truncates input at 256 tokens.
  - **mxbai-embed-large (Ollama).** Pros: stronger on benchmarks; 1024-dim. Cons: a **512-token input limit** caps chunk size, and it's larger and slower.
- **Decision:** `nomic-embed-text` through Ollama's embed API.
- **Why:** One runtime for both models keeps the stack small, and its input limit comfortably covers 400-token chunks plus heading prefixes.
- **Trade-offs accepted:** If we ever change the embedding model, the whole index must be rebuilt, because vectors from different models aren't comparable. We'll store the model name in the index metadata and refuse to load a mismatched index.
- **Likely reviewer questions:**
  - *"Why different prefixes for queries and documents?"* nomic was trained with asymmetric task prefixes. A short question and a long passage get mapped into a shared space more accurately when the model knows which is which.
  - *"Why cosine similarity?"* It compares direction and ignores length. We L2-normalize vectors once at ingest, so cosine becomes a plain dot product.

### D-008: Vector store: NumPy brute-force
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **NumPy.** An `N×768` float32 matrix plus a parallel list of chunk metadata. Pros: exact search; ~30 lines; no extra dependencies; easy to reason about and test. Cons: O(N) per query; we write persistence ourselves; no metadata filtering.
  - **ChromaDB.** Pros: persistence, metadata filtering, a real DB API. Cons: a dependency plus an abstraction this scale doesn't need.
  - **FAISS.** Pros: industry-standard ANN with an obvious scaling story. Cons: a native dependency, and at our scale brute force is already exact and instant.
  - **SQLite + sqlite-vec.** Pros: a single file, SQL filtering. Cons: an extension to install, and still more machinery than we need.
- **Decision:** NumPy, exact cosine search over normalized vectors (`scores = M @ q`, then `argsort`).
- **Why:** At hundreds to thousands of chunks, a matrix multiply takes well under a millisecond. Exact results also make the retrieval eval (D-019) deterministic.
- **Trade-offs accepted:** Memory is N × 768 × 4 bytes (100K chunks ≈ 300 MB), and search grows linearly.
- **Likely reviewer questions:**
  - *"How does this scale?"* Linearly. It's fine up to roughly 10⁵ chunks. Beyond that, FAISS IVF/HNSW trades a little recall for sub-linear search. The store sits behind a small interface (`add`, `search`, `save`, `load`), so swapping it is local to one module.

---

## Context budget (the 6K constraint)

### D-009: Chat history: token-bounded sliding window
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **Sliding window.** Keep the newest turns that fit in ≤ 1,024 tokens and drop the oldest first. Pros: simple, predictable, testable. Cons: older context is lost completely.
  - **Stateless.** Pros: maximum room for retrieved context. Cons: breaks follow-up questions, and this is billed as a chat app.
  - **Rolling summary.** Pros: keeps long-range memory. Cons: an extra LLM call per turn (slow on a laptop); summaries can drop or distort facts; another prompt to budget.
- **Decision:** Sliding window over whole (user, assistant) turn pairs, capped at 1,024 tokens, counting template overhead.
- **Why:** Short demo conversations fit easily, and the behaviour is deterministic and easy to explain.
- **Trade-offs accepted:** A long answer (up to 1,024 tokens) can push all earlier turns out of the window. If that shows up in testing, we can store a truncated version of long assistant answers in history.
- **Likely reviewer questions:**
  - *"What if a single turn exceeds the history budget?"* It gets dropped, or truncated if we add that. History is the first thing to give way, because the documents are the point of RAG.

### D-010: Token counting: Qwen2.5's tokenizer, run locally
- **Status:** accepted — deep-dive pending
- **Context:** Ollama has no public tokenize endpoint, and overflow is silent (D-002), so we need exact counts *before* sending.
- **Options considered:**
  - **Exact.** Load Qwen2.5's `tokenizer.json` with the Hugging Face `tokenizers` library (a Rust core, lightweight, no PyTorch). Pros: exact counts. Cons: one dependency; a one-time download at setup (it then runs offline, like the model weights); tied to the model family.
  - **Heuristic (chars ÷ 4 + margin).** Pros: no dependencies. Cons: badly wrong for code, tables and non-English text (Chinese is roughly one token per character), which can push the prompt past 6K.
  - **Measure afterwards (`prompt_eval_count`).** Pros: ground truth from Ollama. Cons: only detects overflow after it happens.
- **Decision:** The exact tokenizer as the guard, with `prompt_eval_count` logged as a cross-check in debug mode.
- **Why:** Exact counts make the 6K guarantee provable rather than approximate.
- **Subtlety:** Ollama wraps each message in Qwen's ChatML template (`<|im_start|>role\n … <|im_end|>\n`), which adds a few tokens per message. Our counter must add this overhead per message, and the 200-token margin absorbs any remaining error. We'll measure the overhead against `prompt_eval_count` **[verify]**.
- **Likely reviewer questions:**
  - *"Is downloading a tokenizer still 'fully local'?"* Yes. It's a static file fetched once at setup, the same as the model weights. Nothing goes over the network at runtime.
  - *"How do you know your count matches Ollama's?"* We compare our count with `prompt_eval_count` on real requests, and a test fails if they drift beyond the margin.

### D-011: Budget allocation: fixed caps
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **Fixed caps** (see the table at the top). Pros: predictable, easy to explain and unit-test. Cons: any unused history allowance is wasted rather than given to context.
  - **Dynamic, context first.** Pros: uses every token. Cons: history can vanish entirely on queries with many relevant chunks, and behaviour is harder to predict.
  - **Dynamic, history first.** Pros: good for long chats. Cons: starves retrieval, and documents are the point of a RAG app.
- **Decision:** Fixed caps: answer 1,024 · system ~300 · history ≤ 1,024 · margin ~200 · context ≈ 3,596.
- **Why:** Each slot has a clear owner and a hard limit. That makes "how do you stay under 6K?" a one-line answer, and the property easy to test.
- **Trade-offs accepted:** We leave tokens unused when history is short. A natural refinement, if needed, is to let context borrow unused history tokens.
- **Likely reviewer questions:**
  - *"Why reserve 1,024 tokens for the answer?"* In llama.cpp, generated tokens occupy the same context window as the prompt. If the prompt took all 6,144 tokens, the model would have no room to answer. 1,024 tokens is roughly 700 words, plenty for a grounded answer.
  - *"Why does context get the largest share?"* Answer quality in RAG depends mostly on whether the right passage is in the prompt.

---

## Retrieval

### D-012: Ingestion: a separate `ingest` command with an on-disk index
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **`ingest` command writing `index/vectors.npy` + `index/chunks.json`.** Pros: separates indexing from querying; chat starts instantly; the index can be inspected. Cons: the index can go stale if files change and nobody re-runs ingest.
  - **Rebuild on every start.** Pros: never stale; no files. Cons: re-embeds everything on every launch, so the demo starts with a wait.
  - **Incremental (content hashes).** Pros: fast re-ingest. Cons: change tracking and deletion handling for a corpus that rebuilds in seconds.
- **Decision:** A full rebuild on `ingest`, saved to `index/`. The metadata records the embedding model, chunking parameters and creation time.
- **Why:** It's the standard offline-index/online-query split, and a small corpus makes full rebuilds cheap.
- **Trade-offs accepted:** Staleness. Mitigation: `chat` warns if any source file is newer than the index.
- **Likely reviewer questions:**
  - *"What happens if I edit a file?"* Re-run `ingest`, and chat warns you if you forget. Incremental indexing by content hash is the upgrade path.

### D-013: Retrieval query: previous user question + current question
- **Status:** accepted — deep-dive pending
- **Context:** A follow-up like "what about its limitations?" has no subject, so embedding it alone retrieves noise.
- **Options considered:**
  - **Previous + current question.** Pros: no extra LLM call; resolves most pronoun follow-ups. Cons: when the user switches topic, the old topic biases retrieval.
  - **Raw current question.** Pros: simplest. Cons: fails on follow-ups, which reviewers will try.
  - **LLM rewrites the question to stand alone.** Pros: most accurate. Cons: an extra LLM call (~1–3 s), another prompt to budget, and the rewrite can itself go wrong.
- **Decision:** Embed `previous_user_question + "\n" + current_question` (on the first turn, just the current question).
- **Why:** A cheap heuristic that covers the common case.
- **Trade-offs accepted:** Topic switches get a slightly blurred query. If the eval or demo shows this hurts, we upgrade to LLM rewriting.
- **Likely reviewer questions:**
  - *"Why not use the whole history?"* Older turns increasingly blur the query. The previous question alone carries most of the reference.

### D-014: Chunking: structure-aware, ~400 tokens, ~50 overlap, heading-path prefix
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **Structure-aware.** Split on Markdown headings, then paragraphs, then sentences, packing pieces up to ~400 tokens. Overlap of ~50 tokens applies only when a section is cut into several chunks. Pros: chunks are coherent units. The heading prefix (`Guide > Install > macOS`) restores context a chunk would otherwise lose. Code blocks and tables stay whole where possible. Cons: more code; edge cases (headings inside code fences, very long paragraphs).
  - **Fixed token windows.** Pros: simplest. Cons: cuts mid-sentence, mid-code-block and mid-table, and chunks lose their section context.
  - **Smaller (~200 tokens).** Pros: more precise matches, and about 15 fit in the budget. Cons: each chunk has less surrounding context for the LLM.
- **Decision:** Structure-aware, ~400 tokens with ~50 overlap. `.txt` files have no headings, so they fall back to paragraph/sentence splitting with the filename as the prefix.
- **Why:** About 400 tokens is big enough to hold a full explanation and small enough that 6–8 chunks fit in about 3.6K, which gives retrieval several chances to hit.
- **Trade-offs accepted:** 400 is a starting point, not a measured optimum. We'll test 200 vs 400 vs 600 with the retrieval eval (D-019).
- **Likely reviewer questions:**
  - *"Why overlap?"* So a fact that straddles a cut point appears whole in at least one chunk.
  - *"Why prefix the heading path?"* A chunk under `## Limitations` doesn't say *what* it's describing the limitations of. The prefix gets embedded with the text, so the chunk can be retrieved by topic.

### D-015: Chunk selection: fill the context budget best-first
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **Fill by score until ~3,596 tokens (at most 8 chunks).** Pros: uses the budget fully and adapts to chunk lengths. Cons: low-relevance chunks can get in when only a couple are truly relevant.
  - **Fixed k = 4.** Pros: simplest to explain. Cons: wastes budget on short chunks, or needs an overflow check for long ones anyway.
  - **Over-fetch 20 + cross-encoder rerank.** Pros: better precision. Cons: a second model plus PyTorch, overkill for a small corpus.
- **Decision:** Rank by cosine score and add chunks best-first while they fit. A chunk that doesn't fit is skipped and the next one tried, up to 8.
- **Why:** The budget, not an arbitrary k, is the real constraint, and the token counter we already have provides the stopping rule.
- **Trade-offs accepted:** No reranking. Weakly related chunks can add noise. The grounding prompt (D-016) tells the model to ignore irrelevant context.
- **Likely reviewer questions:**
  - *"What about 'lost in the middle'?"* LLMs pay most attention to the start and end of the context. With only 6–8 chunks the effect is small, and putting the best chunk closest to the question is a cheap fix. (Chunk order within the prompt is a detail we'll settle while building.)

### D-016: No relevant match: grounding by prompt only
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **Prompt only.** Always pass the best chunks, and instruct the model to answer only from the context and otherwise say it doesn't know. Pros: no threshold to tune. Cons: a 7B model sometimes answers from its own knowledge anyway.
  - **Similarity threshold → refuse.** Pros: deterministic. Cons: cosine thresholds are specific to the model and corpus, and set too high, they refuse good questions.
  - **Threshold → answer but flag it.** Pros: friendly. Cons: blurs the RAG guarantee.
- **Decision:** Grounding by prompt only.
- **Why:** The simplest approach with no tuning. The fictional `.txt` docs (D-022) give us a direct test of whether the model sticks to context.
- **Trade-offs accepted:** Occasional answers from outside the documents. If testing shows it's frequent, we add a threshold calibrated on the eval set.
- **Likely reviewer questions:**
  - *"How do you stop hallucinations?"* We reduce them: a strict system prompt, sources shown under each answer, and fictional docs to test grounding. Prompting can't fully prevent them, and a score threshold is the next layer.

### D-017: Sources: a list printed after each answer
- **Status:** accepted — deep-dive pending
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
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **`rag_chat/` package** (`config`, `loaders`, `chunker`, `tokens`, `embedder`, `store`, `retriever`, `prompt`, `llm`, `cli`). Pros: each stage testable in isolation; the review walkthrough follows the data flow. Cons: more files.
  - **Single script.** Pros: fastest start. Cons: hard to test and hard to walk through.
  - **Ports/adapters, with interfaces for the LLM, embedder and store.** Pros: shows the system can swap components. Cons: abstractions with a single implementation are code to defend without benefit.
- **Decision:** The package, using plain functions and small classes, with no abstract interfaces until a second implementation exists.
- **Likely reviewer questions:**
  - *"How would you swap Ollama for llama.cpp?"* Only `llm.py`, `embedder.py` and (for counting) `tokens.py` know about Ollama, so the change stays inside those modules.

### D-019: Testing: pytest unit tests + retrieval eval
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **Unit tests + retrieval eval.** Pros: proves the deterministic properties hold, and gives evidence for tuning choices. Cons: writing about 15 eval questions takes time.
  - **Unit tests only.** Pros: less work. Cons: chunk size and k stay guesses.
  - **Also an LLM-judged answer eval.** Pros: measures answer faithfulness. Cons: slow and noisy with a 7B judge; a big extra build.
- **Decision:** pytest for the chunker, tokenizer budget (**the assembled prompt never exceeds 6,144 − 1,024**), history window and store. Plus `eval/questions.jsonl` with about 15 questions, each labelled with its expected source file, scored as **hit-rate@k**.
- **Why:** Gives evidence for the claim that we stay under 6K and for the chunk size.
- **Likely reviewer questions:**
  - *"How did you pick 400 tokens?"* We ran the eval at 200, 400 and 600 and compared hit-rates. The numbers go here once we have them.

### D-020: Streaming output
- **Status:** accepted — deep-dive pending
- **Decision:** Stream tokens from Ollama (`stream: true`) and print them as they arrive. Sources print after the stream ends.
- **Why:** A 300–1,000-token answer at ~25 tok/s takes 12–40 s. Without streaming the demo shows a frozen terminal. Streaming also improves time to the first token, which is how users perceive latency.
- **Trade-offs accepted:** Slightly more complex code: handling partial chunks, Ctrl-C mid-stream, and errors mid-stream.

### D-021: Setup: venv + pinned `requirements.txt`
- **Status:** accepted — deep-dive pending
- **Options considered:**
  - **venv + pinned requirements.** Pros: standard, needs no extra tools. Cons: no lockfile for transitive dependencies (fine with so few dependencies).
  - **pyproject + uv.** Pros: fast and reproducible. Cons: reviewers need uv installed.
  - **Docker.** Pros: reproducible. Cons: **Docker on macOS can't use the Metal GPU**, so Ollama in a container would run on CPU only and be far too slow.
- **Decision:** venv + pinned `requirements.txt`. Setup steps: install Ollama → `ollama pull qwen2.5:7b-instruct nomic-embed-text` → venv + `pip install` → a one-time tokenizer download → `ingest` → `chat`.

### D-022: Demo corpus: public Markdown docs + fictional `.txt` notes
- **Status:** accepted — deep-dive pending (revised: originally public docs only)
- **Context:** The demo must show both `.md` **and** `.txt` ingestion, and must prove that answers come from retrieval rather than from the model's training data.
- **Options considered:**
  - **Public docs only.** Pros: realistic Markdown structure. Cons: no `.txt` files; Qwen may already know these docs, so a correct answer doesn't prove retrieval worked.
  - **Fictional only.** Pros: the model can't know the answers. Cons: writing 20+ realistic files takes a lot of time.
  - **Mixed.** Pros: covers both formats plus a grounding proof. Cons: a little more work.
- **Decision:** A subset of permissively licensed open-source Markdown docs (tens of files with headings, code blocks and tables), **plus a few fictional `.txt` documents**, such as internal notes for a made-up company or product, containing specific facts the model cannot have seen.
- **Why:** The fictional docs give a decisive demo moment: ask a question only they can answer, get the right answer, and see the `.txt` source listed. Asking something in neither set shows the model saying it doesn't know.
- **Open details:** which public docs to use (check the licence and attribute it in the README), and the content of the fictional files and the eval questions.
- **Likely reviewer questions:**
  - *"How do I know the model isn't just answering from pretraining?"* Ask about the fictional company. Those facts exist only in our `.txt` files.
