import numpy as np

from rag_chat import config, embedder
from rag_chat.chunker import Chunk

def query_vector(current: str, previous: str | None) -> np.ndarray: # (768,),
    # weighting current question and previous questions
    q_curr = embedder.embed_query(current)
    if previous is None:            # first turn
        return q_curr
    q_prev = embedder.embed_query(previous)
    q = config.QUERY_WEIGHT * q_curr + (1 - config.QUERY_WEIGHT) * q_prev
    return q / np.linalg.norm(q)    # renormalize so cosine scores stay comparable


def select(ranked: list[tuple[Chunk, float]]) -> list[(Chunk, float)]:
    # relevance only: no token counting here (prompt.py fits the budget)
    if not ranked:
        return []
    top = ranked[0][1]                     # ranked is already sorted best-first
    cutoff = max(config.SCORE_FLOOR, top - config.SCORE_DELTA)
    kept = [(c, s) for (c, s) in ranked if s >= cutoff]
    return kept[: config.MAX_CHUNKS]