import os
import tempfile

import gigatoken as gt

from tokn.cli import count_files, parse_exts, resolve_tokenizer, _should_list_file


# ---------------------------------------------------------------------------
# parse_exts tests
# ---------------------------------------------------------------------------


def test_parse_exts_empty():
    assert parse_exts([]) is None


def test_parse_exts_with_dots():
    assert parse_exts([".py", ".md"]) == {".py", ".md"}


def test_parse_exts_without_dots():
    assert parse_exts(["py", "md"]) == {".py", ".md"}


def test_parse_exts_mixed():
    assert parse_exts([".py", "js"]) == {".py", ".js"}


# ---------------------------------------------------------------------------
# _should_list_file tests
# ---------------------------------------------------------------------------


def test_should_list_file_no_filters():
    # Hidden file should be skipped
    assert not _should_list_file(
        "/tmp/.hidden",
        skip_hidden=True,
        include_exts=None,
        exclude_exts=None,
    )
    # Normal file should be included
    assert _should_list_file(
        "/tmp/file.py",
        skip_hidden=True,
        include_exts=None,
        exclude_exts=None,
    )
    # Hidden file with skip_hidden=False should be included
    assert _should_list_file(
        "/tmp/.hidden",
        skip_hidden=False,
        include_exts=None,
        exclude_exts=None,
    )


def test_should_list_file_include_ext():
    include = {".py", ".md"}
    assert _should_list_file(
        "file.py",
        skip_hidden=True,
        include_exts=include,
        exclude_exts=None,
    )
    assert not _should_list_file(
        "file.txt",
        skip_hidden=True,
        include_exts=include,
        exclude_exts=None,
    )


def test_should_list_file_exclude_ext():
    exclude = {".pyc", ".so"}
    assert _should_list_file(
        "file.py",
        skip_hidden=True,
        include_exts=None,
        exclude_exts=exclude,
    )
    assert not _should_list_file(
        "file.pyc",
        skip_hidden=True,
        include_exts=None,
        exclude_exts=exclude,
    )


# ---------------------------------------------------------------------------
# count_files tests
# ---------------------------------------------------------------------------


def test_count_files_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "hello.py")
        with open(fp, "w", encoding="utf-8") as f:
            f.write("def hello():\n    return 'world'\n")
        tokenizer = gt.Tokenizer("openai-community/gpt2")
        counts, skipped = count_files([fp], tokenizer)
        assert counts[0] > 0
        assert not skipped[0]


def test_count_files_binary():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "data.bin")
        # Write non-UTF-8 bytes
        with open(fp, "wb") as f:
            f.write(bytes(range(256)))
        tokenizer = gt.Tokenizer("openai-community/gpt2")
        counts, skipped = count_files([fp], tokenizer)
        assert counts[0] == 0
        assert skipped[0]


def test_count_files_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "empty.txt")
        with open(fp, "w", encoding="utf-8") as f:
            f.write("")
        tokenizer = gt.Tokenizer("openai-community/gpt2")
        counts, skipped = count_files([fp], tokenizer)
        assert counts[0] == 0
        assert not skipped[0]


# ---------------------------------------------------------------------------
# resolve_tokenizer tests
# ---------------------------------------------------------------------------


def test_resolve_tokenizer_valid():
    tok = resolve_tokenizer("gpt-4o")
    assert isinstance(tok, gt.Tokenizer)


def test_resolve_tokenizer_invalid_fallback():
    tok = resolve_tokenizer("nonexistent-model-xyz-12345")
    assert isinstance(tok, gt.Tokenizer)
