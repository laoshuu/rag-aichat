"""Exact Qwen2.5 token counting (D-010). The only module that knows the tokenizer.

Two public functions, used by two different callers:
  count(text)              -> chunker (n_tokens), prompt builder (pieces)
  count_messages(messages) -> prompt builder (final guarantee), llm (drift check)
"""

from tokenizers import Tokenizer
from rag_chat import config
import math

# Ollama wraps each message in Qwen's ChatML template
OVERHEAD_PER_MSG = 5  # per message
TRAILER = 3           # trailing <|im_start|>assistant\n

# --- Load once ---------------------------------------------------------------
if not config.TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Tokenizer file not found: {config.TOKENIZER_PATH}. "
        "Run `python -m rag_chat.download_tokenizer` first."
    )

_tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH)) # importing this module fails if the file is missing


def count(text: str) -> int:
    """Number of Qwen tokens in `text` (content only, no chat-template overhead)."""
    # Qwen's tokenizer adds no BOS/EOS either way, but being explicit means the count can't change if the post-processor does.
    return len(_tokenizer.encode(text, add_special_tokens=False).ids)


def count_messages(messages: list[dict]) -> int:
    """Tokens Ollama will process for these chat messages (prompt_eval_count).

    Σ content + OVERHEAD_PER_MSG per message + TRAILER; matched Ollama exactly in M0.
    """
    content = sum(count(m["content"]) for m in messages)
    return content + len(messages) * OVERHEAD_PER_MSG + TRAILER
  
def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    enc = _tokenizer.encode(text, add_special_tokens=False)
    cuts = [0] + [enc.offsets[i][0] for i in range(max_tokens, len(enc.ids), max_tokens)] + [len(text)]
    return [text[a:b] for a, b in zip(cuts, cuts[1:]) if text[a:b]]   # skip empty slices