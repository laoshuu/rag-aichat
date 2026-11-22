"""Find and read the knowledge base: .md / .txt files -> Document(path, text).

No retrieval, no chunking, no embeddings. Read the files safely, nothing else.
Caller: the `ingest` command (load -> chunk -> embed -> store). Chat never uses it.
Policy for bad files: D-023 (skip + warn, never silent, never fatal).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"} 


@dataclass(frozen=True)
class Document:
    path: str  # relative to the ingest root; becomes Chunk.source (D-017)
    text: str


def load_documents(root: Path) -> list[Document]:
    """Recursively load every .md/.txt file under `root`, sorted by path.

    Unreadable (non-UTF-8) and empty files are skipped with a warning on stderr.
    A missing or non-directory `root` raises: that's a typo, not one bad file.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Knowledge base path is not a directory: {root}")

    # Recursive search on files
    files = sorted(
        f
        for f in root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_SUFFIXES
        and not _is_hidden(f.relative_to(root))
    )

    documents = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        try:
            # utf-8-sig strips a BOM; text mode turns \r\n and \r into \n.
            text = f.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            _warn(f"skipped {rel}: not valid UTF-8")
            continue
        if not text.strip():
            _warn(f"skipped {rel}: empty")
            continue
        documents.append(Document(path=rel, text=text))
    return documents


def _is_hidden(rel: Path) -> bool:
    """True for dotfiles or anything inside a dot-folder (.git/, macOS ._guide.md)."""
    return any(part.startswith(".") for part in rel.parts)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)
