from rag_chat import config, tokens
from rag_chat.chunker import Chunk

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.
If the context does not contain the answer, say you don't know based on the documents.
Be concise."""    # must stay ≤ config.SYSTEM_PROMPT_LENGTH

PROMPT_LIMIT = config.TOT_CONTEXT_LENGTH - config.ANS_LENGTH      # 5,120: the hard limit

def _system(text):
    return {"role": "system", "content": text}

def _user(text):
    return {"role": "user", "content": text}

def _assistant(text):
    return {"role": "assistant", "content": text}

def window(history: list[tuple[str, str]]) -> list[str]:
    # D-009: newest whole pairs that fit in HISTORY_LENGTH, template overhead included
    kept, used = [], 0
    for (q, a) in reversed(history):              # newest first
        cost = tokens.count(q) + tokens.count(a) + 2 * 5     # 2 messages × 5 overhead
        if used + cost > config.HISTORY_LENGTH:
            break                                 # stop, don't skip: no gaps in the conversation
        kept = [_user(q), _assistant(a)] + kept
        used += cost
    return kept


def query_message(chunks: list[Chunk], current: str) -> str:
    if not chunks:
        context = "(no relevant documents found)" # triggers the "say you don't know" rule (D-016)
    else:
        context = "\n\n".join(f"[{i}] {c.source}\n{c.full_text}" for i, c in enumerate(chunks, 1))
    return f"Context:\n{context}\n\nQuestion: {current}"


def build(history, ranked: list[(Chunk, float)], current: str) -> (list[dict], list[(Chunk, float)], int):
    system  = _system(SYSTEM_PROMPT)
    hist    = window(history)

    # question alone must fit the context slot
    if tokens.count(query_message([], current)) > config.RETRIEVAL_LENGTH:
        raise ValueError("question too long")

    # fit chunks best-first; skip the ones that don't fit, try the next
    included = []
    for (c, s) in ranked:
        candidate = included + [(c, s)]
        if tokens.count(query_message([c for c, _ in candidate], current)) <= config.RETRIEVAL_LENGTH:
            included = candidate

    messages = [system, *hist, _user(query_message([c for c, _ in included], current))]
    expected = tokens.count_messages(messages)
    assert expected <= PROMPT_LIMIT              # the guarantee; should never fire
    return messages, included, expected