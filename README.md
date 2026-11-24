# RAG Chat

A command-line chat app that answers questions from a folder of `.md` and `.txt` files. Everything runs locally: the LLM (`qwen2.5:7b-instruct`), the embeddings (`nomic-embed-text`), the vector store (NumPy) and retrieval. The whole prompt fits a **6,144-token context window**, checked with exact token counts.

## Quick start

Needs macOS or Linux, Python 3.11+, and about 6 GB free for the models.

```bash
# 1. Ollama (https://ollama.com/download), then the two models
ollama serve &                       # skip if the Ollama app is already running
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# 2. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m rag_chat.download_tokenizer     # one time: pinned Qwen tokenizer -> models/

# 3. Build the index, then chat
python -m rag_chat ingest data/
python -m rag_chat chat
```

Other commands:

```bash
python -m rag_chat search "How do I deploy to GitHub Pages?"   # ranked chunks, no LLM
python -m pytest -q                                             # unit tests, no Ollama needed
python -m eval.run_eval                                         # retrieval eval + parameter sweeps
```

In `chat`: `/reset` clears the history, `/quit` exits, and Ctrl-C stops the current answer.

## Things to try

| Ask | What it shows |
|---|---|
| *When is the Lumen 4.0 launch and why did it move?* → *Who is fixing it?* | Answer from a fictional `.txt` file the model can't have seen, then a pronoun follow-up |
| *How do I add a custom CSS file to my MkDocs site?* | Answer from the public Markdown docs, with code |
| *How many devices can a Free account sync?* | A fact read from a Markdown table |
| *How do I configure Kubernetes autoscaling?* | Nothing relevant: the model says the documents don't cover it |

## How it works

```
INGEST   .md/.txt ─► loader ─► chunker ─► embedder ─► index/ (vectors.npy, chunks.json, manifest.json)
                               heading-aware,  nomic-embed-text,
                               ≤400 tokens     unit vectors

CHAT     question ─► query vector = 0.8·current + 0.2·previous question
                 ─► cosine search over all chunks ─► keep score ≥ 0.60 and ≥ top − 0.10 (max 8)
                 ─► prompt = system + recent history + chunks that fit the budget + question
                 ─► qwen2.5:7b-instruct (num_ctx 6144), streamed ─► answer + source list
```

| Slot in the 6,144-token window | Tokens |
|---|---|
| Answer (`num_predict`) | 1,024 |
| System prompt | ≤ 300 |
| Chat history (newest whole turns) | ≤ 1,024 |
| Margin | 50 |
| Retrieved chunks + question | ≤ 3,746 |
| **Prompt hard limit** | **≤ 5,120**, checked on the exact token count of every request |

One module per stage in `rag_chat/`: `config`, `tokens`, `loader`, `chunker`, `embedder`, `store`, `retriever`, `prompt`, `llm`, `cli`. There is no RAG framework. Every design decision, the options considered and the measured numbers are in [docs/decisions.md](docs/decisions.md).

## Retrieval eval

`eval/questions.jsonl` holds 15 short conversations (22 turns): single questions, follow-ups, topic switches and unanswerable questions. `run_eval.py` runs the real pipeline in memory and reports hit-rate@k. These results set the parameters:

| Chunk size | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| 200 | 0.78 | 0.94 | 1.00 |
| **400** | **0.94** | **1.00** | **1.00** |
| 600 | 0.89 | 1.00 | 1.00 |

Weighting the current question at w = 0.8 gave the best follow-up hit@1 (1.00, vs 0.60 at w = 0.7, 0.9 and 1.0). Floor 0.60 with δ 0.10 kept the right chunk for every answerable turn while sending about 5.7 chunks instead of 7.2.

## Corpus and licence

- `data/mkdocs/`: 13 pages of the [MkDocs](https://github.com/mkdocs/mkdocs) 1.6.1 documentation, © 2014-present Tom Christie, BSD-2-Clause (see `data/mkdocs/LICENSE`). Pages that are only build directives, plus the changelog, are left out.
- `data/lumen_guide.md`, `data/support_playbook.txt`, `data/team_notes.txt`: a fictional product written for this project. They test whether answers come from retrieval rather than from what the model already knows.

To use your own documents, run `python -m rag_chat ingest <folder>`. If a file changes after ingest, `chat` warns you to re-run it.
