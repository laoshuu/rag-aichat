"""Tests for rag_chat.store (D-008, D-012). No Ollama: hand-made vectors."""
import json
import os

import numpy as np
import pytest

from rag_chat import config, store
from rag_chat.chunker import Chunk

# Three chunks, three one-hot vectors: the "right answer" for any query is obvious.
CHUNKS = [
    Chunk("lumen_guide.md", ["Lumen Notes Guide", "Sync"], "Lumen syncs every 5 minutes.", 12),
    Chunk("lumen_guide.md", ["Lumen Notes Guide", "Export"], "Export to PDF or Markdown.", 11),
    Chunk("team_notes.txt", ["team_notes.txt"], "Launch moved to May 12.", 9),
]
VECTORS = np.eye(3, dtype=np.float32)        # row i = [0..1..0], norm 1


@pytest.fixture
def saved(tmp_path):
    index_dir = tmp_path / "index"
    store.save(index_dir, CHUNKS, VECTORS, tmp_path)
    return index_dir


# --- save -> load ------------------------------------------------------------

def test_round_trip_keeps_chunks_vectors_and_manifest(saved):
    index = store.load(saved)
    assert index.chunks == CHUNKS                 # heading_path comes back as a list
    np.testing.assert_array_equal(index.vectors, VECTORS)
    assert index.vectors.dtype == np.float32
    assert index.manifest["embed_model"] == config.EMBED_MODEL
    assert index.manifest["chunk_size"] == config.CHUNK_SIZE
    assert index.manifest["overlap"] == config.OVERLAP
    assert index.manifest["n_chunks"] == 3


def test_save_rejects_mismatched_lengths_and_writes_nothing(tmp_path):
    index_dir = tmp_path / "index"
    with pytest.raises(ValueError):
        store.save(index_dir, CHUNKS, VECTORS[:2], tmp_path)
    assert not index_dir.exists()


# --- search ------------------------------------------------------------------

def test_known_vector_ranks_first_and_all_chunks_returned(saved):
    index = store.load(saved)
    q = np.array([0.0, 1.0, 0.0], dtype=np.float32)     # points exactly at chunk 1
    results = index.search(q)
    assert [c for c, _ in results][0] == CHUNKS[1]
    assert len(results) == 3                            # ALL chunks: D-015 filters later
    assert [s for _, s in results] == [1.0, 0.0, 0.0]
    assert all(type(s) is float for _, s in results)    # plain floats, not np.float32


def test_scores_are_sorted_best_first(saved):
    index = store.load(saved)
    q = np.array([0.6, 0.0, 0.8], dtype=np.float32)     # mostly chunk 2, some chunk 0
    results = index.search(q)
    assert [c.source for c, _ in results] == ["team_notes.txt", "lumen_guide.md", "lumen_guide.md"]
    assert [round(s, 3) for _, s in results] == [0.8, 0.6, 0.0]


# --- the three load checks ---------------------------------------------------

def test_missing_index_says_run_ingest(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        store.load(tmp_path / "nope")


def test_different_embed_model_is_refused(saved, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", "mxbai-embed-large")
    with pytest.raises(RuntimeError, match="ingest"):
        store.load(saved)


def test_different_chunk_params_only_warn(saved, monkeypatch, capsys):
    monkeypatch.setattr(config, "CHUNK_SIZE", 600)
    index = store.load(saved)                           # still loads
    assert len(index.chunks) == 3
    assert "warning" in capsys.readouterr().err


def test_vector_chunk_count_mismatch_raises(saved):
    np.save(saved / "vectors.npy", VECTORS[:2])         # simulate a half-written index
    with pytest.raises(RuntimeError, match="ingest"):
        store.load(saved)


def test_index_files_are_inspectable_json(saved):
    chunks = json.loads((saved / "chunks.json").read_text(encoding="utf-8"))
    assert chunks[0]["heading_path"] == ["Lumen Notes Guide", "Sync"]


def test_source_edited_after_ingest_warns(saved, tmp_path, capsys):
    store.load(saved)
    assert "changed" not in capsys.readouterr().err     # fresh index: quiet
    doc = tmp_path / "notes.md"
    doc.write_text("# edited later")
    later = (saved / "manifest.json").stat().st_mtime + 10
    os.utime(doc, (later, later))
    store.load(saved)                                   # still loads, but warns
    assert "re-run ingest" in capsys.readouterr().err
