#!/usr/bin/env python3
"""
tokn — print GPT token counts for files/directories or stdin using gigatoken.

Usage examples:
  tokn                      # count tokens in current directory (ALL file types)
  tokn file.md dir/         # count tokens in specific paths
  echo "text" | tokn        # count tokens from stdin
  tokn --per-file           # show per-file counts plus total
  tokn --json               # JSON output
  tokn --model gpt-4o       # select model/resolve tokenizer

Dependencies: pip install gigatoken
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from importlib.metadata import version as _pkg_version

try:
    import gigatoken as gt
except ImportError:
    print(
        "gigatoken is required. Install with: pip install gigatoken",
        file=sys.stderr,
    )
    sys.exit(2)

_VERSION = f"tokn {_pkg_version('tokn')} (gigatoken {_pkg_version('gigatoken')})"


# ---------------------------------------------------------------------------
# Model / tokenizer resolution
# ---------------------------------------------------------------------------

# Map common model names to known HuggingFace repos for fallback resolution.
# Vocab sizes: o200k_base=200k, cl100k_base=100k, gpt2=50k.
_MODEL_FALLBACK_MAP: dict[str, str] = {
    "gpt-4o": "Xenova/gpt-4o",
    "gpt-4o-mini": "Xenova/gpt-4o",
    "gpt-4": "Xenova/gpt-4",
    "gpt-4-turbo": "Xenova/gpt-4",
    "gpt-4-32k": "Xenova/gpt-4",
    "gpt-3.5-turbo": "Xenova/gpt-3.5-turbo",
    "gpt-3.5-turbo-16k": "Xenova/gpt-3.5-turbo",
    "text-embedding-3-large": "Xenova/gpt-3.5-turbo",
    "text-embedding-3-small": "Xenova/gpt-3.5-turbo",
    "text-embedding-ada-002": "Xenova/gpt-3.5-turbo",
    "gpt-5o": "Xenova/gpt-4o",
}

_UNIVERSAL_FALLBACK = "openai-community/gpt2"


def resolve_tokenizer(model: str) -> gt.Tokenizer:
    """Resolve a gigatoken Tokenizer for the given model name.

    Strategy:
      1. Try ``gt.Tokenizer(model)`` directly (HF repo, path, etc.).
      2. Map to a known HuggingFace repo for the model family.
      3. Fall back to ``"openai-community/gpt2"`` with a warning on stderr.
    """
    # 1 – direct
    try:
        return gt.Tokenizer(model)
    except Exception:
        pass

    # 2 – known HF repo mapping
    hf_repo = _MODEL_FALLBACK_MAP.get(model)
    if hf_repo is not None:
        try:
            return gt.Tokenizer(hf_repo)
        except Exception:
            pass

    # 3 – universal fallback
    print(
        f"Warning: could not resolve tokenizer for model '{model}', "
        f"falling back to {_UNIVERSAL_FALLBACK}",
        file=sys.stderr,
    )
    try:
        return gt.Tokenizer(_UNIVERSAL_FALLBACK)
    except Exception:
        print(
            "Error: could not create any tokenizer. Is gigatoken installed correctly?",
            file=sys.stderr,
        )
        sys.exit(3)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# Directories always skipped during traversal (build artifacts, VCS, etc.).
# Exact names are matched; entries ending with a dot-prefixed suffix
# (e.g. ".egg-info") also match any directory whose name ends with it.
_ALWAYS_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__",
    ".git",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
})

_ALWAYS_SKIP_SUFFIXES: tuple[str, ...] = (
    ".egg-info",
)


def _should_skip_dir(dirname: str) -> bool:
    """Return True if *dirname* should be pruned from traversal."""
    if dirname in _ALWAYS_SKIP_DIRS:
        return True
    return dirname.endswith(_ALWAYS_SKIP_SUFFIXES)

def _should_list_file(
    fp: str,
    *,
    skip_hidden: bool,
    include_exts: set[str] | None,
    exclude_exts: set[str] | None,
) -> bool:
    """Return True if *fp* should be included in the token-count walk."""
    basename = os.path.basename(fp)

    if skip_hidden and basename.startswith("."):
        return False

    ext = os.path.splitext(fp)[1].lower()

    if include_exts is not None and ext not in include_exts:
        return False

    if exclude_exts is not None and ext in exclude_exts:
        return False

    return True


def iter_files(
    paths: list[str],
    *,
    skip_hidden: bool = True,
    include_exts: set[str] | None = None,
    exclude_exts: set[str] | None = None,
):
    """Yield every readable file path found under each *paths* entry."""
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: path does not exist: {p}", file=sys.stderr)
            continue

        if os.path.isfile(p):
            if _should_list_file(
                p,
                skip_hidden=skip_hidden,
                include_exts=include_exts,
                exclude_exts=exclude_exts,
            ):
                yield p

        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                # Prune build artifacts and (optionally) hidden directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not _should_skip_dir(d)
                    and not (skip_hidden and d.startswith("."))
                ]
                for name in files:
                    fp = os.path.join(root, name)
                    if _should_list_file(
                        fp,
                        skip_hidden=skip_hidden,
                        include_exts=include_exts,
                        exclude_exts=exclude_exts,
                    ):
                        yield fp


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------

def parse_exts(ext_list: list[str]) -> set[str] | None:
    """Normalise a list of extension strings into a set (lower, dot-prefixed).

    Returns ``None`` when the list is empty (meaning "no filter").
    """
    if not ext_list:
        return None
    parsed: set[str] = set()
    for raw in ext_list:
        val = raw.strip() if raw.startswith(".") else f".{raw.strip()}"
        parsed.add(val.lower())
    return parsed


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_files(
    file_paths: list[str],
    tokenizer: gt.Tokenizer,
) -> tuple[list[int], list[bool]]:
    """Count tokens per file using batch encoding for speed.

    Returns a tuple of (counts, skipped) where counts[i] is the token count
    for file_paths[i], and skipped[i] is True if the file was unreadable/binary.
    """
    texts: list[str] = []
    skipped: list[bool] = []
    warned = False

    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8", errors="strict") as fh:
                texts.append(fh.read())
            skipped.append(False)
        except (UnicodeDecodeError, OSError):
            if not warned:
                print(
                    f"Warning: skipping binary/unreadable file "
                    f"(e.g. {os.path.basename(fp)})",
                    file=sys.stderr,
                )
                warned = True
            texts.append("")
            skipped.append(True)

    # Batch encode all at once for speed
    results = tokenizer.encode_batch(texts)
    # encode_batch returns a list of token ID lists
    counts = [len(r) for r in results]

    return counts, skipped


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(n: int) -> str:
    return f"{n:,}"


def print_results(
    file_paths: list[str],
    counts: list[int],
    skipped: list[bool],
    total: int,
    *,
    per_file: bool,
    as_json: bool,
) -> None:
    """Emit results in the requested format."""
    if as_json:
        files_data = [
            {"path": p, "tokens": c, "skipped": s}
            for p, c, s in zip(file_paths, counts, skipped)
        ]
        payload = {"files": files_data, "total": total}
        print(json.dumps(payload, indent=2))
        return

    if per_file:
        for p, c in zip(file_paths, counts):
            print(f"{_fmt(c):>10}  {p}")
        print(f"{'─' * 10}")

    print(_fmt(total))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Count GPT tokens in files/directories or stdin using gigatoken. "
            "By default reads ALL file types in the current directory "
            "(or the given paths)."
        ),
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Files and/or directories. Defaults to current dir (TTY) or stdin (piped).",
    )
    ap.add_argument(
        "--ext",
        "-e",
        action="append",
        default=[],
        help="Include ONLY files with this extension (repeatable). Without this, ALL types are included.",
    )
    ap.add_argument(
        "--exclude-ext",
        action="append",
        default=[],
        help="Exclude files with this extension (repeatable).",
    )
    ap.add_argument(
        "--hidden",
        action="store_true",
        help="Include hidden files and directories.",
    )
    ap.add_argument(
        "--model",
        "-m",
        default="gpt-4o",
        help="Model name to resolve tokenizer (default: gpt-4o).",
    )
    ap.add_argument(
        "--per-file",
        action="store_true",
        help="Show per-file token counts before the total.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    ap.add_argument(
        "--version",
        action="version",
        version=_VERSION,
    )

    args = ap.parse_args()

    # -- Resolve tokenizer ---------------------------------------------------
    tokenizer = resolve_tokenizer(args.model)

    # -- Extension filters ---------------------------------------------------
    include_exts = parse_exts(args.ext)
    exclude_exts = parse_exts(args.exclude_ext)

    # -- Determine input source ----------------------------------------------
    paths: list[str] = args.paths

    if not paths:
        # Read stdin only when it's actually a pipe (FIFO) with data
        # piped in — not just because stdin is a non-TTY
        # (e.g. CI, shell subshells, IDE terminals).
        stdin_is_pipe = False
        if not sys.stdin.isatty():
            try:
                stdin_is_pipe = stat.S_ISFIFO(os.fstat(0).st_mode)
            except OSError:
                pass

        if stdin_is_pipe:
            text = sys.stdin.read()
            total = len(tokenizer.encode(text))

            if args.json:
                payload = {
                    "files": [{"path": "<stdin>", "tokens": total, "skipped": False}],
                    "total": total,
                }
                print(json.dumps(payload, indent=2))
            else:
                print(_fmt(total))
            return

        # Default to current working directory
        paths = ["."]

    # -- Collect file paths --------------------------------------------------
    file_paths = list(
        iter_files(
            paths,
            skip_hidden=not args.hidden,
            include_exts=include_exts,
            exclude_exts=exclude_exts,
        )
    )

    # -- No files found ------------------------------------------------------
    if not file_paths:
        print("No files found.", file=sys.stderr)
        return

    # -- Count tokens --------------------------------------------------------
    counts, skipped = count_files(file_paths, tokenizer)

    total = sum(counts)

    # -- Output --------------------------------------------------------------
    print_results(
        file_paths,
        counts,
        skipped,
        total,
        per_file=args.per_file,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()
