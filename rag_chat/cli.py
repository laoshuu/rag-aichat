"""Command line: `python -m rag_chat ingest <dir>` | `search "<question>"` (D-005, D-012).
Thin layer: parse args, call the pipeline modules in order, print, map errors to exit 1.
No logic of its own; `chat` joins in M3."""

import argparse
import sys
import time
from pathlib import Path
from rag_chat import config, loader, chunker, embedder, store

# ---------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    t0   = time.perf_counter()
    docs = loader.load_documents(Path(args.data_dir))        # D-023 warnings go to stderr
    chunks = [c for d in docs for c in chunker.chunk(d)]
    if not chunks:
        raise RuntimeError(f"No .md/.txt content found under {args.data_dir}.")
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    vectors = embedder.embed_documents([c.full_text for c in chunks])    # full_text: prefix+body
    store.save(config.INDEX_DIR, chunks, vectors)               # only after ALL embeds succeed
    print(f"Index written to {config.INDEX_DIR} in {time.perf_counter() - t0:.1f}s")
    return 0

# ---------------------------------------------------------------- search (no LLM)
def cmd_search(args) -> int:
    index   = store.load(config.INDEX_DIR)            # 3 checks happen here
    q       = embedder.embed_query(args.question)
    results = index.search(q)[: args.k]
    for rank, (c, score) in enumerate(results, 1):
        print(f"{rank}. {score:.3f}  {label(c)}")
        print(" " + c.text[:100].replace("\n", " "))
    return 0

def label(c: chunker.Chunk) -> str:                   # "lumen_guide.md › Lumen Notes Guide › Sync"
    if c.heading_path == [c.source]:                  # .txt: heading_path is just the filename
        return c.source                               # avoid "notes.txt › notes.txt"
    return " > ".join([c.source] + c.heading_path)

# ---------------------------------------------------------------- entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rag_chat")
    sub    = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="build index/ from a folder of .md/.txt")
    p.add_argument("data_dir")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="show ranked chunks for a question (no LLM)")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, NotADirectoryError) as e:   # expected user-facing errors
        print(f"error: {e}", file=sys.stderr)
        return 1