"""Text -> unit vectors via Ollama's nomic-embed-text (D-007).
Callers: ingest (embed_documents on Chunk.full_text), search/chat (embed_query)."""

import numpy as np
import requests
from rag_chat import config

DOC_PREFIX   = "search_document: "
QUERY_PREFIX = "search_query: "

# ---------------------------------------------------------------- private
def _post(texts: list[str]) -> np.ndarray:          # shape (len(texts), 768)
    try:
        resp = requests.post(f"{config.OLLAMA_BASE_URL}/api/embed",
                             json={"model": config.EMBED_MODEL, "input": texts},
                             timeout=60)
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama not reachable at {config.OLLAMA_BASE_URL}. Run `ollama serve`.")
    if resp.status_code == 404:                          # verified: unknown model -> 404
        raise RuntimeError(f"Model {config.EMBED_MODEL} not found. Run `ollama pull {config.EMBED_MODEL}`.")
    resp.raise_for_status()                              # anything else: let it be loud
    return np.array(resp.json()["embeddings"], dtype=np.float32)

def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)  # Always normalize, in case embed model changed

# ---------------------------------------------------------------- public
def embed_documents(texts: list[str]) -> np.ndarray:   # (N, 768), each row norm 1
    batches = []
    N = len(texts)
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE): 
        batch = [DOC_PREFIX + t for t in texts[start : start + config.EMBED_BATCH_SIZE]]
        batches.append(_post(batch))
        print(f"embedded {min(start+config.EMBED_BATCH_SIZE, N)}/{N}")
    return _normalize(np.vstack(batches))

def embed_query(text: str) -> np.ndarray:              # (768,), norm 1
    return _normalize(_post([QUERY_PREFIX + text]))[0]