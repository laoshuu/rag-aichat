"""Split one Document into heading-aware ~400-token Chunks (D-014).

Pure function: Document in, list[Chunk] out. No disk, no Ollama, no embeddings.
Caller: the `ingest` command, once per document (load -> CHUNK -> embed -> store).

    text ──► 1. scan lines ──► sections (heading_path + blocks)
                                   │
                                   ▼
             2. break any block > budget into sentences, then token pieces
                                   │
                                   ▼
             3. pack pieces greedily into chunks (+ sentence overlap)
                                   │
                                   ▼
             4. wrap as Chunk(source, heading_path, text, n_tokens)
"""

import re
from dataclasses import dataclass, replace
from typing import NamedTuple

from rag_chat import config
from rag_chat.loader import Document
from rag_chat.tokens import count, split_by_tokens


# =============================================================================
# The output shape
# =============================================================================
@dataclass(frozen=True)
class Chunk:
    source: str              # Document.path
    heading_path: list[str]  # ["Lumen Notes Guide", "Sync"]
    text: str                # body only
    n_tokens: int            # count(full_text): prefix + body, Qwen tokens

    @property
    def full_text(self) -> str:      # what gets embedded AND put in the prompt
        return " > ".join(self.heading_path) + "\n" + self.text
    
@dataclass(frozen=True)
class Section:
    heading_path: list[str]
    blocks: list[str] # one paragraph OR one whole fenced code block


# =============================================================================
# 1. Scan lines -> sections. ONE pass, two pieces of state.
# =============================================================================
HEADING = re.compile(r"^(#{1,6})\s+(.*)")   # 1-6 '#', then whitespace, then title
FENCE   = re.compile(r"^(```|~~~)")

def _scan(text: str, fallback_path: str, markdown: bool = True) -> list[Section]:
    """Split text into sections (one per heading) of blocks (paragraphs / code).

    markdown=False (.txt files): no heading detection, so a line a person typed
    as "# action items" stays text. The whole file is one [fallback_path] section.
    """
    stack = []           # current heading path
    fence = None         # marker of the open fence ("```" or "~~~"), None outside
    lines = []           # lines within a block
    blocks = []          # multiple blocks belong to a section
    result = []          # list of sections to be returned
    
    def end_block():     # blank line, fences, heading, end of file
        nonlocal lines
        if lines:        # also stops empty blocks from appearing
            blocks.append("\n".join(lines))
        lines = []

    def end_section():   # heading, end of file
        nonlocal blocks
        end_block()
        if blocks:       # drops headings with no text (## Install)
            result.append(Section(stack or [fallback_path], blocks))
        blocks = []

    for line in text.split("\n"):
        if fence is not None:                 # inside a code block
            lines.append(line)                # '#' comments, blank lines: just code
            if line.startswith(fence):        # only the SAME marker closes it
                end_block()                   # the code block is one unit
                fence = None
        elif m := FENCE.match(line):          # opening fence
            end_block()                       # don't glue the paragraph above onto it
            lines.append(line)
            fence = m.group(1)
        elif markdown and (m := HEADING.match(line)):
            end_section()                     # BEFORE updating stack: old path
            level = len(m.group(1))
            stack = stack[:level - 1] + [m.group(2).strip()]  # pop to level-1, push
        elif not line.strip():                # blank line = paragraph boundary
            end_block()
        else:
            lines.append(line)
    end_section()                             # last block/section (or unclosed fence)
    return result

# =============================================================================
# 2. Make every piece fit the budget.
# =============================================================================
# The ~400 target includes the prefix (A2), so the BODY budget per section is:
#     budget = config.CHUNK_SIZE - count(prefix + "\n")

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Piece(NamedTuple):
    """One unit for _pack. Says WHAT it is; _join decides how to format it."""
    text: str
    starts_block: bool  # first piece of a paragraph/code block (else continues one)
    code: bool          # came from a fenced code block (never used as overlap)


def _split_sentences(text: str) -> list[str]:
    """Lossless sentence split: each sentence keeps its trailing whitespace,
    so "".join(result) == text (paragraph line breaks survive)."""
    out, start = [], 0
    for m in SENTENCE_END.finditer(text):
        out.append(text[start:m.end()])
        start = m.end()
    out.append(text[start:])
    return [s for s in out if s]


def _pieces(block: str, budget: int, code: bool = False) -> list[str]:
    """Break one block into parts that each fit `budget`. "".join(parts) == block."""
    if count(block) <= budget:
        return [block]                           # common case: block fits
    if code:
        # Code has no prose sentences. Known limit: token cuts can land mid-line
        # and leave a fence unbalanced. Deferred (D-014): no code block in our
        # corpus exceeds the budget; re-check when the public doc set is added.
        return split_by_tokens(block, budget)
    result = []
    for s in _split_sentences(block):
        if count(s) <= budget:
            result.append(s)
        else:
            result += split_by_tokens(s, budget) # Q8: last resort (e.g. a big table)
    return result


# =============================================================================
# 3. Pack pieces into chunks, greedy, within ONE section.
# =============================================================================
def _join(text: str, piece: Piece) -> str:
    """Append a piece to a chunk being built."""
    if not text:
        return piece.text
    if piece.starts_block:                       # new block: paragraph break
        return text.rstrip() + "\n\n" + piece.text
    return text + piece.text                     # same block: text already has its space


def _overlap(text: str, limit: int) -> str:
    """Whole trailing sentences of `text` totalling <= limit tokens (Q10).

    Never the first sentence, so the overlap is never the entire previous chunk.
    Returns "" if even the last sentence is over the limit.
    """
    tail = ""
    for s in reversed(_split_sentences(text)[1:]):
        if count(s + tail) > limit:
            break
        tail = s + tail
    return tail


def _pack(pieces: list[Piece], budget: int) -> list[str]:
    """Greedily glue pieces into chunks of <= budget tokens, with sentence overlap.

    Called once per section, so overlap never crosses a heading (D-014).
    Counting: recount the whole candidate each time (exact; counts aren't additive,
    A2). Quadratic in pieces per section, which is fine at our corpus size.
    """
    chunks = []
    text = ""            # the chunk being built
    last_was_code = False
    for p in pieces:
        candidate = _join(text, p)
        if text and count(candidate) > budget:      # p doesn't fit: close the chunk
            chunks.append(text.strip())
            # Seed the next chunk with the end of the one just closed...
            text = "" if last_was_code else _overlap(text, config.OVERLAP)
            candidate = _join(text, p)
            # start clean if overlap + p cannot fits
            if text and count(candidate) > budget:
                candidate = p.text
        text = candidate
        last_was_code = p.code
    # pack remaining text
    if text.strip():
        chunks.append(text.strip())
    return chunks


# =============================================================================
# 4. Public function
# =============================================================================
def chunk(doc: Document) -> list[Chunk]:
    """Document -> heading-aware chunks of ~CHUNK_SIZE tokens (prefix included)."""
    result = []
    is_md = doc.path.lower().endswith(".md")
    for section in _scan(doc.text, doc.path, markdown=is_md):
        prefix = " > ".join(section.heading_path)
        budget = config.CHUNK_SIZE - count(prefix + "\n")
        pieces = []
        for block in section.blocks:
            code = bool(FENCE.match(block))
            for i, part in enumerate(_pieces(block, budget, code)):
                pieces.append(Piece(part, starts_block=(i == 0), code=code))
        for body in _pack(pieces, budget):
            c = Chunk(doc.path, section.heading_path, body, n_tokens=0)
            result.append(replace(c, n_tokens=count(c.full_text)))  # exact, final
    return result