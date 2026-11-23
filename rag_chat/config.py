"""All constants for the app. Constants only: no logic, no I/O (D-018)."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Anchor paths to the repo root, not the current working directory, so
# `python -m rag_chat ...` works no matter where it is run from.
ROOT = Path(__file__).resolve().parent.parent
OLLAMA_BASE_URL = "http://localhost:11434"
INDEX_DIR = ROOT / "index"

# --- Context budget (D-011) --------------------------------------------------
TOT_CONTEXT_LENGTH = 6144
ANS_LENGTH = 1024
HISTORY_LENGTH = 1024
SYSTEM_PROMPT_LENGTH = 300
MARGIN = 50
RETRIEVAL_LENGTH = TOT_CONTEXT_LENGTH - ANS_LENGTH - HISTORY_LENGTH - SYSTEM_PROMPT_LENGTH - MARGIN


# --- Chunking (D-014) --------------------------------------------------------
CHUNK_SIZE = 400    # token length of each chunk
OVERLAP = 50        # max size of whole-sentence overlap

# --- Models (D-006, D-007) ---------------------------------------------------
CHAT_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text:latest"

# Tokenizer: a pinned local copy, loaded with Tokenizer.from_file (D-010).
# Fetched once by `python -m rag_chat.download_tokenizer`; models/ is gitignored.
TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"  # HF commit
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKENIZER_PATH = ROOT / "models" / "qwen2.5-tokenizer.json"


# --- Embeddings ---------------------------------------------------
EMBED_BATCH_SIZE = 64
