"""Retrieval eval (D-019): hit-rate@k plus sweeps of chunk size, w, floor and δ.

    python -m eval.run_eval            # needs Ollama running (embeddings only, no LLM)

Runs the real pipeline (loader -> chunker -> embedder -> Index -> retriever) in memory,
so index/ is never touched. Each turn uses the previous question in its conversation,
exactly like `chat` (D-013).
"""
import json
from pathlib import Path

from rag_chat import chunker, config, embedder, loader, retriever
from rag_chat.store import Index

QUESTIONS = Path(__file__).parent / "questions.jsonl"
KS = (1, 3, 5)


def load_turns() -> list[dict]:
    turns = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        conv, prev = json.loads(line), None
        for t in conv["turns"]:
            turns.append({"id": conv["id"], "q": t["q"], "prev": prev, "expect": t["expect"]})
            prev = t["q"]
    return turns


def matches(chunk, expect: list[str]) -> bool:
    # "file" or "file#heading substring" (needed when one file holds several topics)
    for e in expect:
        source, _, heading = e.partition("#")
        if chunk.source == source and heading in " > ".join(chunk.heading_path):
            return True
    return False


def build_index(docs, chunk_size: int) -> Index:
    config.CHUNK_SIZE = chunk_size               # chunker reads it at call time
    chunks = [c for d in docs for c in chunker.chunk(d)]
    return Index(chunks, embedder.embed_documents([c.full_text for c in chunks]), {})


def rank_all(index: Index, turns: list[dict], w: float) -> list[list]:
    config.QUERY_WEIGHT = w
    return [index.search(retriever.query_vector(t["q"], t["prev"])) for t in turns]


def hit_rates(turns, ranked) -> dict[int, float]:
    pairs = [(t, r) for t, r in zip(turns, ranked) if t["expect"]]
    return {k: sum(any(matches(c, t["expect"]) for c, _ in r[:k]) for t, r in pairs) / len(pairs) for k in KS}


def selection(turns, ranked, floor: float, delta: float) -> tuple[float, float, float]:
    config.SCORE_FLOOR, config.SCORE_DELTA = floor, delta
    kept = [retriever.select(r) for r in ranked]
    ans  = [(t, k) for t, k in zip(turns, kept) if t["expect"]]
    una  = [k for t, k in zip(turns, kept) if not t["expect"]]
    recall = sum(any(matches(c, t["expect"]) for c, _ in k) for t, k in ans) / len(ans)
    n_kept = sum(len(k) for _, k in ans) / len(ans)
    empty  = sum(not k for k in una) / len(una)
    return recall, n_kept, empty


def main() -> None:
    turns = load_turns()
    docs  = loader.load_documents(config.ROOT / "data")
    base  = (config.CHUNK_SIZE, config.QUERY_WEIGHT, config.SCORE_FLOOR, config.SCORE_DELTA)
    print(f"{len(turns)} turns ({sum(bool(t['expect']) for t in turns)} answerable)\n")

    print("chunk_size  hit@1  hit@3  hit@5")
    for size in (200, 400, 600):
        h = hit_rates(turns, rank_all(build_index(docs, size), turns, base[1]))
        print(f"{size:>10}  {h[1]:.2f}   {h[3]:.2f}   {h[5]:.2f}")

    index = build_index(docs, base[0])
    print("\n   w  hit@1  hit@3  hit@5   (follow-up turns only)")
    follow = [t for t in turns if t["prev"] and t["expect"]]
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        h = hit_rates(follow, rank_all(index, follow, w))
        print(f"{w:.1f}  {h[1]:.2f}   {h[3]:.2f}   {h[5]:.2f}")

    ranked = rank_all(index, turns, base[1])
    print("\nTop score per turn (sets the floor):")
    for t, r in zip(turns, ranked):
        tag = "ANS" if t["expect"] else "UNA"
        rank = next((i for i, (c, _) in enumerate(r, 1) if matches(c, t["expect"])), "-")
        print(f"  {tag} {r[0][1]:.3f}  first hit @{rank:<3} {t['q']}")

    print("\nfloor  delta  recall  avg_kept  unanswerable_empty")
    for floor in (0.50, 0.60, 0.64, 0.65, 0.66, 0.68):
        for delta in (0.05, 0.10, 0.15, 0.20):
            rec, n, empty = selection(turns, ranked, floor, delta)
            print(f"{floor:.2f}   {delta:.2f}   {rec:.2f}    {n:4.1f}      {empty:.2f}")

    config.CHUNK_SIZE, config.QUERY_WEIGHT, config.SCORE_FLOOR, config.SCORE_DELTA = base


if __name__ == "__main__":
    main()
