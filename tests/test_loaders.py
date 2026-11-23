"""Tests for rag_chat.loader (D-023: skip + warn, never silent, never fatal)."""
import pytest

from rag_chat.loader import Document, load_documents


def write(root, rel, data):
    """Create root/rel with `data` (str -> UTF-8 text, bytes -> raw)."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def paths(docs):
    return [d.path for d in docs]


# --- Which files are found ---------------------------------------------------

def test_finds_md_and_txt_and_ignores_other_types(tmp_path):
    write(tmp_path, "guide.md", "# Guide\n")
    write(tmp_path, "notes.txt", "Launch moved to May 12.\n")
    write(tmp_path, "slides.pdf", "not text we support\n")
    write(tmp_path, "script.py", "print('hi')\n")
    assert paths(load_documents(tmp_path)) == ["guide.md", "notes.txt"]


def test_finds_files_in_nested_folders_with_relative_paths(tmp_path):
    # Real doc sets are nested; two index.md files must stay distinguishable.
    write(tmp_path, "docs/tutorial/index.md", "# Tutorial\n")
    write(tmp_path, "docs/advanced/index.md", "# Advanced\n")
    assert paths(load_documents(tmp_path)) == [
        "docs/advanced/index.md",
        "docs/tutorial/index.md",
    ]


def test_upper_case_suffix_is_found(tmp_path):
    write(tmp_path, "GUIDE.MD", "# Guide\n")
    assert paths(load_documents(tmp_path)) == ["GUIDE.MD"]


def test_hidden_files_and_folders_are_skipped(tmp_path):
    write(tmp_path, "guide.md", "# Guide\n")
    write(tmp_path, ".git/notes.md", "# internal\n")
    write(tmp_path, "._guide.md", b"\x00\x05\x16\x07 macOS resource fork")
    assert paths(load_documents(tmp_path)) == ["guide.md"]


def test_output_is_sorted_by_path(tmp_path):
    # Created out of order; ingest must be deterministic regardless.
    for name in ["c.md", "a.md", "b.txt"]:
        write(tmp_path, name, f"# {name}\n")
    assert paths(load_documents(tmp_path)) == ["a.md", "b.txt", "c.md"]


def test_returns_document_with_full_text(tmp_path):
    write(tmp_path, "guide.md", "# Guide\n\nHello.\n")
    assert load_documents(tmp_path) == [Document(path="guide.md", text="# Guide\n\nHello.\n")]


# --- Bad files: skipped, reported, ingest continues (D-023) ------------------

def test_empty_and_whitespace_only_files_are_skipped_with_warning(tmp_path, capsys):
    write(tmp_path, "empty.md", "")
    write(tmp_path, "blank.txt", "  \n\n\t\n")
    write(tmp_path, "good.md", "# Good\n")
    assert paths(load_documents(tmp_path)) == ["good.md"]
    err = capsys.readouterr().err
    assert "empty.md" in err
    assert "blank.txt" in err


def test_non_utf8_file_is_skipped_with_warning_and_ingest_continues(tmp_path, capsys):
    write(tmp_path, "latin1.txt", "café".encode("latin-1"))  # 0xE9: invalid UTF-8
    write(tmp_path, "good.md", "# Good\n")
    assert paths(load_documents(tmp_path)) == ["good.md"]
    assert "latin1.txt" in capsys.readouterr().err


def test_warnings_go_to_stderr_not_stdout(tmp_path, capsys):
    write(tmp_path, "empty.md", "")
    load_documents(tmp_path)
    out, err = capsys.readouterr()
    assert out == ""
    assert "empty.md" in err


# --- Text normalisation ------------------------------------------------------

def test_utf8_bom_is_stripped(tmp_path):
    write(tmp_path, "bom.md", b"\xef\xbb\xbf# Title\n")
    (doc,) = load_documents(tmp_path)
    assert doc.text == "# Title\n"  # no leading ﻿, so "# Title" is still a heading


def test_crlf_and_cr_line_endings_become_lf(tmp_path):
    # Guards the read_text() behaviour: switching to read_bytes() would break this.
    write(tmp_path, "windows.md", b"# Guide\r\n## Sync\r\nOn.\r\n")
    write(tmp_path, "oldmac.txt", b"line one\rline two\r")
    docs = {d.path: d.text for d in load_documents(tmp_path)}
    assert docs["windows.md"] == "# Guide\n## Sync\nOn.\n"
    assert docs["oldmac.txt"] == "line one\nline two\n"


# --- Bad root: fatal ---------------------------------------------------------

def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_file_instead_of_directory_raises(tmp_path):
    f = write(tmp_path, "guide.md", "# Guide\n")
    with pytest.raises(NotADirectoryError):
        load_documents(f)


def test_empty_directory_returns_empty_list(tmp_path):
    assert load_documents(tmp_path) == []
