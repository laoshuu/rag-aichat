"""Persist and search the index: vectors.npy + chunks.json + manifest.json (D-008, D-012).
Callers: ingest (save), search/chat (load -> Index.search)."""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from pathlib import Path
from rag_chat import config
from rag_chat.chunker import Chunk

# ---------------------------------------------------------------- the loaded index
@dataclass(frozen=True)
class Index:
    chunks:   list[Chunk]       # row i of vectors  <->  chunks[i]
    vectors:  np.ndarray        # (N, 768) float32, rows norm 1
    manifest: dict

    def search(self, query_vector: np.ndarray) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vector      # (N,) cosine, both sides unit length
        order  = np.argsort(-scores)              # best first; ALL of them (filters later)
        return [(self.chunks[i], float(scores[i])) for i in order]

# ---------------------------------------------------------------- write
def save(dir: Path, chunks: list[Chunk], vectors: np.ndarray, data_dir: Path) -> None:
    if len(chunks) != len(vectors):    # caller bug: fail before writing anything
        raise ValueError(f"Length Mismatch: {len(chunks)} chunks but {len(vectors)} vectors")
    dir.mkdir(parents=True, exist_ok=True)
    np.save(dir / "vectors.npy", vectors)
    with (dir / "chunks.json").open('w', encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    with (dir / "manifest.json").open('w', encoding="utf-8") as f:
        json.dump({
            "embed_model": config.EMBED_MODEL,
            "chunk_size":  config.CHUNK_SIZE,
            "overlap":     config.OVERLAP,
            "n_chunks":    len(chunks),
            "data_dir":    str(data_dir.resolve()),     # for the staleness warning on load
            "created_at":  datetime.now().isoformat(),
        }, f, indent=2)

# ---------------------------------------------------------------- read + 3 checks
def load(dir: Path) -> Index:
    if not (dir / "manifest.json").exists():
        raise FileNotFoundError(f"No index at {dir}. Run `python -m rag_chat ingest data/`.")
    manifest = json.loads((dir / "manifest.json").read_text(encoding="utf-8"))

    # 1. REFUSE: different model = different vector space
    if manifest["embed_model"] != config.EMBED_MODEL:
        raise RuntimeError(f"Index built with {manifest['embed_model']}, config says {config.EMBED_MODEL}. Re-run ingest.")

    # 2. WARN: stale chunking, but search is still correct
    if (manifest["chunk_size"], manifest["overlap"]) != (config.CHUNK_SIZE, config.OVERLAP):
        print(f"warning: index chunked with chunk_size = {manifest['chunk_size']}, overlap = {manifest['overlap']}; re-run ingest to apply", file=sys.stderr)

    # 2b. WARN: a source file changed after ingest (D-012 staleness)
    built = (dir / "manifest.json").stat().st_mtime
    data  = Path(manifest.get("data_dir", ""))
    if data.is_dir() and any(f.stat().st_mtime > built for f in data.rglob("*") if f.suffix in (".md", ".txt")):
        print(f"warning: files in {data} changed after the index was built; re-run ingest", file=sys.stderr)

    vectors = np.load(dir / "vectors.npy")        # allow_pickle stays False (default)
    chunks  = [Chunk(**d) for d in json.loads((dir / "chunks.json").read_text(encoding="utf-8"))]

    # 3. RAISE: row i must be chunk i
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Index corrupt: {len(vectors)} vectors vs {len(chunks)} chunks. Re-run ingest.")

    return Index(chunks, vectors, manifest)