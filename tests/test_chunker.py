"""Tests for rag_chat.chunker (D-014).

Stage 1 (_scan): text -> sections (one per heading) of blocks (paragraphs / code).
Stage 2 (_pieces): oversized blocks -> sentences -> token pieces.
Stage 3 (_pack): pieces -> chunks <= budget, with whole-sentence overlap.
Stage 4 (chunk): Document -> list[Chunk].
Fixtures live in tests/fixtures/chunker/; the realistic doc is data/lumen_guide.md.
"""
from pathlib import Path

import pytest

from rag_chat import config
from rag_chat.chunker import (
    Piece, Section, _overlap, _pack, _pieces, _scan, _split_sentences, chunk,
)
from rag_chat.loader import Document
from rag_chat.tokens import count

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "chunker"
LNG = "Lumen Notes Guide"


def scan_file(path: Path) -> list[Section]:
    """Scan a file the way chunk() will: headings only for .md."""
    return _scan(path.read_text(encoding="utf-8"), path.name,
                 markdown=path.suffix.lower() == ".md")


def paths(sections):
    return [s.heading_path for s in sections]


# =============================================================================
# _scan: the realistic doc
# =============================================================================

def test_lumen_guide_sections_and_block_counts():
    sections = scan_file(ROOT / "data" / "lumen_guide.md")
    assert [(s.heading_path, len(s.blocks)) for s in sections] == [
        (["lumen_guide.md"], 1),                          # preamble -> filename
        ([LNG], 1),
        ([LNG, "Install", "macOS"], 2),                   # "## Install" alone: dropped
        ([LNG, "Install", "Linux"], 2),
        ([LNG, "Install", "Windows"], 1),
        ([LNG, "Sync"], 1),
        ([LNG, "Sync", "Devices and limits"], 2),
        ([LNG, "Sync", "Conflicts"], 7),
        ([LNG, "Shortcuts"], 1),
        ([LNG, "Pricing"], 1),
    ]


# =============================================================================
# _scan: headings
# =============================================================================

def test_heading_stack_pops_to_parent_level():
    assert paths(scan_file(FIXTURES / "heading_pop.md")) == [
        ["H1", "A", "A1"],
        ["H1", "B"],          # "## B" after "### A1": pop A and A1
        ["Second H1"],        # "# Second H1": pop everything
    ]


def test_heading_with_no_text_produces_no_section():
    assert paths(scan_file(FIXTURES / "empty_section.md")) == [["H1", "Has text"]]


def test_heading_lines_are_not_part_of_any_block():
    (section,) = _scan("# Title\nBody.\n", "x.md")
    assert section.blocks == ["Body."]


def test_hashtag_and_seven_hashes_are_not_headings():
    sections = scan_file(FIXTURES / "not_headings.md")
    assert paths(sections) == [["not_headings.md"], ["Real heading"]]
    assert sections[0].blocks == [
        "#hashtag is not a heading\n####### seven hashes is not a heading"
    ]


def test_heading_title_whitespace_is_stripped():
    # \s+ accepts a tab or several spaces; the title must not keep them.
    assert paths(_scan("#\tTabbed\nx\n##   Spaced  \ny\n", "x.md")) == [
        ["Tabbed"], ["Tabbed", "Spaced"]
    ]


# =============================================================================
# _scan: fallback path (no heading above the text)
# =============================================================================

def test_preamble_before_first_heading_gets_filename():
    sections = scan_file(FIXTURES / "preamble.md")
    assert paths(sections) == [["preamble.md"], ["Title"]]
    assert sections[0].blocks == ["Text before any heading."]


def test_markdown_without_headings_is_one_filename_section():
    (section,) = scan_file(FIXTURES / "no_headings.md")
    assert section.heading_path == ["no_headings.md"]
    assert section.blocks == ["Just a paragraph.", "Another paragraph, no headings at all."]


def test_txt_does_not_detect_headings():
    # A person typed "# action items" in a plain-text note: it's text, not a heading.
    (section,) = scan_file(FIXTURES / "txt_with_hash.txt")
    assert section.heading_path == ["txt_with_hash.txt"]
    assert section.blocks[1].startswith("# action items")


def test_real_txt_file_is_one_section_of_paragraphs():
    (section,) = scan_file(ROOT / "data" / "team_notes.txt")
    assert section.heading_path == ["team_notes.txt"]
    assert len(section.blocks) == 7


# =============================================================================
# _scan: paragraphs (blocks)
# =============================================================================

def test_blank_line_separates_blocks_single_newline_does_not():
    (section,) = _scan("# S\nline one\nline two\n\nnext para\n", "x.md")
    assert section.blocks == ["line one\nline two", "next para"]


def test_whitespace_only_line_is_a_paragraph_boundary():
    (section,) = _scan("# S\nfirst\n   \nsecond\n", "x.md")
    assert section.blocks == ["first", "second"]


def test_table_is_one_block():
    (section,) = _scan("# S\n| a | b |\n|---|---|\n| 1 | 2 |\n", "x.md")
    assert section.blocks == ["| a | b |\n|---|---|\n| 1 | 2 |"]


def test_no_empty_blocks_from_repeated_or_trailing_blank_lines():
    (section,) = _scan("# S\n\n\n\npara\n\n\n", "x.md")
    assert section.blocks == ["para"]


@pytest.mark.parametrize("text", ["", "\n\n", "   \n\t\n"])
def test_empty_or_whitespace_text_gives_no_sections(text):
    assert _scan(text, "x.md") == []


# =============================================================================
# _scan: code fences
# =============================================================================

def test_hash_comment_inside_fence_is_not_a_heading():
    # Roadmap "done when" case.
    text = "# Setup\nRun:\n\n```bash\n# install deps\npip install lumen\n```\n"
    (section,) = _scan(text, "x.md")
    assert section.heading_path == ["Setup"]
    assert section.blocks == ["Run:", "```bash\n# install deps\npip install lumen\n```"]


def test_blank_line_inside_fence_does_not_split_code_block():
    text = "# S\n```python\ndef a():\n    pass\n\ndef b():\n    pass\n```\n"
    (section,) = _scan(text, "x.md")
    assert section.blocks == ["```python\ndef a():\n    pass\n\ndef b():\n    pass\n```"]


def test_opening_fence_ends_the_paragraph_above_it():
    # No blank line between the paragraph and the fence (as in lumen_guide.md macOS).
    text = "# S\nDownload Lumen.\n```bash\nlumen --version\n```\n"
    (section,) = _scan(text, "x.md")
    assert section.blocks == ["Download Lumen.", "```bash\nlumen --version\n```"]


def test_tilde_fence_works_like_backticks():
    text = "# S\n~~~\n# not a heading\n\nstill code\n~~~\n"
    (section,) = _scan(text, "x.md")
    assert section.blocks == ["~~~\n# not a heading\n\nstill code\n~~~"]


def test_fence_closes_only_with_its_own_marker():
    text = "# T\n```md\n~~~ not a close\n# still code\n```\nafter\n"
    (section,) = _scan(text, "x.md")
    assert section.heading_path == ["T"]
    assert section.blocks == ["```md\n~~~ not a close\n# still code\n```", "after"]


def test_unclosed_fence_keeps_rest_of_file_as_code():
    (section,) = scan_file(FIXTURES / "unclosed_fence.md")
    assert section.heading_path == ["Setup"]
    assert len(section.blocks) == 2
    code = section.blocks[1]
    assert code.startswith("```python\n# not a heading")
    assert "def b():" in code


def test_fence_inside_txt_still_groups_lines():
    # Fences are kept in .txt too: harmless, and keeps pasted code together.
    (section,) = _scan("notes\n```\na\n\nb\n```\n", "n.txt", markdown=False)
    assert section.blocks == ["notes", "```\na\n\nb\n```"]


# =============================================================================
# Helpers for stages 2-4
# =============================================================================

SENT = "Lumen keeps both copies of the note."          # one short sentence
PARA = " ".join([SENT] * 12)                            # ~100 tokens, 12 sentences


def piece(text, starts_block=True, code=False):
    return Piece(text, starts_block, code)


# =============================================================================
# _split_sentences
# =============================================================================

def test_split_sentences_is_lossless():
    text = "First one. Second one!\nThird on a new line? Last"
    parts = _split_sentences(text)
    assert parts == ["First one. ", "Second one!\n", "Third on a new line? ", "Last"]
    assert "".join(parts) == text


def test_split_sentences_without_sentence_end_is_one_part():
    assert _split_sentences("| a | b |\n| 1 | 2 |") == ["| a | b |\n| 1 | 2 |"]


# =============================================================================
# _pieces
# =============================================================================

def test_block_that_fits_is_returned_whole():
    assert _pieces(PARA, budget=1000) == [PARA]


def test_oversized_prose_splits_at_sentences_losslessly():
    budget = count(PARA) // 2
    parts = _pieces(PARA, budget)
    assert len(parts) == 12                       # one per sentence (packing is _pack's job)
    assert all(count(p) <= budget for p in parts)
    assert "".join(parts) == PARA


def test_single_sentence_over_budget_falls_back_to_token_split():
    table = "\n".join(f"| E{1000 + i} | Sync request {i} failed |" for i in range(40))
    budget = 50
    parts = _pieces(table, budget)
    assert len(parts) > 1
    assert all(count(p) <= budget + 1 for p in parts)   # +1: counts not exactly additive
    assert "".join(parts) == table


def test_oversized_code_goes_straight_to_token_split():
    # The code has ". " inside; a sentence split would cut there. Code skips it.
    code = "```python\n" + "\n".join(f"x{i} = obj.attr. value  # step {i}" for i in range(40)) + "\n```"
    budget = 60
    parts = _pieces(code, budget, code=True)
    assert len(parts) > 1
    assert all(count(p) <= budget + 1 for p in parts)
    assert "".join(parts) == code


# =============================================================================
# _overlap
# =============================================================================

def test_overlap_takes_whole_trailing_sentences_within_limit():
    text = "Alpha is first. Beta is second. Gamma is third."
    limit = count("Beta is second. Gamma is third.")
    assert _overlap(text, limit) == "Beta is second. Gamma is third."


def test_overlap_is_empty_when_last_sentence_exceeds_limit():
    long_last = "Short. " + "This last sentence is far too long to be copied " * 5 + "here."
    assert _overlap(long_last, limit=10) == ""


def test_overlap_never_copies_the_whole_previous_chunk():
    text = "One. Two."
    assert _overlap(text, limit=1000) == "Two."     # never the first sentence


# =============================================================================
# _pack
# =============================================================================

def test_pieces_that_fit_together_become_one_chunk_without_overlap():
    a, b = "Alpha one. " * 10, "Beta two. " * 10
    chunks = _pack([piece(a.strip()), piece(b.strip())], budget=1000)
    assert chunks == [a.strip() + "\n\n" + b.strip()]


def test_new_block_joins_with_blank_line_continuation_joins_directly():
    chunks = _pack([piece("First sentence. "), piece("Second sentence.", starts_block=False),
                    piece("New paragraph.")], budget=1000)
    assert chunks == ["First sentence. Second sentence.\n\nNew paragraph."]


def test_cut_seeds_next_chunk_with_trailing_sentences():
    sentences = [f"Sentence number {i} is here. " for i in range(30)]
    pieces = [piece(s, starts_block=(i == 0)) for i, s in enumerate(sentences)]
    budget = 120
    chunks = _pack(pieces, budget)
    assert len(chunks) >= 2
    assert all(count(c) <= budget for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = _overlap(prev, config.OVERLAP)
        assert tail                                # every cut here has an overlap
        assert nxt.startswith(tail.strip())        # next chunk begins with it
        assert count(tail) <= config.OVERLAP


def test_overlap_dropped_when_overlap_plus_piece_does_not_fit():
    a = " ".join(["Alpha one."] * 25)    # ~75 tokens, many tiny sentences
    b = " ".join(["Beta two."] * 25)
    chunks = _pack([piece(a), piece(b)], budget=count(b) + 5)
    assert chunks == [a, b]              # b starts clean: overlap would overflow


def test_no_overlap_after_a_code_block():
    code = "```\n" + "print('hi')\n" * 20 + "```"
    prose = " ".join([SENT] * 10)
    chunks = _pack([piece(code, code=True), piece(prose)], budget=count(prose) + 30)
    assert chunks == [code, prose]


def test_pack_of_nothing_is_nothing():
    assert _pack([], budget=100) == []


# =============================================================================
# chunk(): the public function
# =============================================================================

def lumen_chunks():
    text = (ROOT / "data" / "lumen_guide.md").read_text(encoding="utf-8")
    return chunk(Document("lumen_guide.md", text))


def test_lumen_guide_chunk_paths():
    # 10 sections; only Conflicts (489 tokens) is over budget and splits in two.
    assert [c.heading_path for c in lumen_chunks()] == [
        ["lumen_guide.md"],
        [LNG],
        [LNG, "Install", "macOS"],
        [LNG, "Install", "Linux"],
        [LNG, "Install", "Windows"],
        [LNG, "Sync"],
        [LNG, "Sync", "Devices and limits"],
        [LNG, "Sync", "Conflicts"],
        [LNG, "Sync", "Conflicts"],
        [LNG, "Shortcuts"],
        [LNG, "Pricing"],
    ]


def test_conflicts_chunks_overlap_by_the_planned_sentence():
    c1, c2 = [c for c in lumen_chunks() if c.heading_path[-1] == "Conflicts"]
    shared = ("In that case Lumen keeps the larger file and moves the other one "
              "to the Trash, where it stays for 30 days.")
    assert c1.text.endswith(shared)
    assert c2.text.startswith(shared)


def test_no_overlap_across_sections():
    # Each section's first chunk starts with that section's own first block.
    chunks = lumen_chunks()
    sections = scan_file(ROOT / "data" / "lumen_guide.md")
    firsts = [c for i, c in enumerate(chunks)
              if i == 0 or c.heading_path != chunks[i - 1].heading_path]
    assert [c.text.split("\n")[0] for c in firsts] == \
           [s.blocks[0].split("\n")[0] for s in sections]


def test_n_tokens_is_exact_count_of_full_text():
    for c in lumen_chunks():
        assert c.n_tokens == count(c.full_text)


def test_full_text_is_prefix_newline_body():
    c = lumen_chunks()[-1]
    assert c.full_text == "Lumen Notes Guide > Pricing\n" + c.text


def test_source_is_document_path():
    assert {c.source for c in lumen_chunks()} == {"lumen_guide.md"}


def test_every_chunk_fits_chunk_size_including_big_table():
    # Roadmap "done when": a long section splits into several chunks within budget.
    text = (FIXTURES / "big_table.md").read_text(encoding="utf-8")
    chunks = chunk(Document("big_table.md", text))
    assert len(chunks) > 1
    for c in chunks + lumen_chunks():
        assert c.n_tokens <= config.CHUNK_SIZE + 2     # small slack: re-tokenizing cuts


def test_txt_chunk_uses_filename_as_path():
    text = (ROOT / "data" / "team_notes.txt").read_text(encoding="utf-8")
    (c,) = chunk(Document("team_notes.txt", text))
    assert c.heading_path == ["team_notes.txt"]
    assert c.full_text.startswith("team_notes.txt\nLumen team notes")


def test_upper_case_md_suffix_still_detects_headings():
    (c,) = chunk(Document("README.MD", "# Title\nBody.\n"))
    assert c.heading_path == ["Title"]


@pytest.mark.parametrize("text", ["", "\n\n  \n"])
def test_empty_document_gives_no_chunks(text):
    assert chunk(Document("empty.md", text)) == []


def test_chunks_reconstruct_section_text_when_no_cut():
    # A section that fits is one chunk whose body is its blocks joined by blank lines.
    (c,) = chunk(Document("x.md", "# S\nPara one.\n\nPara two.\n"))
    assert c.text == "Para one.\n\nPara two."
