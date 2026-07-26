# tokn

A CLI to easily calculate how many tokens your files take. Because context pollution exists and you need to know *exactly* how much of your precious 200K window that 15,000-line YAML config is about to vaporize.

Powered by [gigatoken](https://github.com/marcelroed/gigatoken) under the hood, because `tiktoken` was built for prompt-sized strings and this thing runs at **24 GB/s** for when you're counting *everything*.

## Install

```bash
pip install tokn
```

Or from source:

```bash
git clone https://github.com/HenryGetz/tokn.git
cd tokn
pip install -e .
```

## Usage

Count everything in the current directory:

```bash
$ tokn
3,756
```

That's it. No renaming files to `.txt` like a caveman. No extension whitelist. Just the number.

Pipe stuff in:

```bash
$ echo "hello world" | tokn
3
```

See what's eating your context budget:

```bash
$ tokn --per-file
       115  pyproject.toml
     2,759  tokn/cli.py
        14  tokn/__init__.py
         9  tokn/__main__.py
       859  tests/test_cli.py
──────────
3,756
```

JSON for the spreadsheet people:

```bash
$ tokn --json
{
  "files": [
    {"path": "src/main.py", "tokens": 1234, "skipped": false}
  ],
  "total": 1234
}
```

## Flags

| Flag | What it does |
|------|-------------|
| `--ext .py` | Only count files with this extension. Repeatable. |
| `--exclude-ext .pyc` | Skip files with this extension. Repeatable. |
| `--hidden` | Count hidden files too. Hope you know what you're doing. |
| `--model gpt-4o` | Use a specific model's tokenizer. Different models, different counts. |
| `--per-file` | Show per-file breakdown. |
| `--json` | Output as JSON. |
| `--version` | `tokn 0.1.0 (gigatoken 0.9.0)` |

## Why not just use `wc -c`?

Because `wc -c` counts bytes, not tokens. And those are *very* different numbers. Go ahead, paste 100KB of Python into ChatGPT and tell me how many tokens that was. I'll wait.

## Why not just use `tiktoken`?

You totally can. `tiktoken` is great for `len(enc.encode("hello world"))`. `gigatoken` (what powers this) is a Rust BPE tokenizer that hits **~24 GB/s** on large files. That's roughly 680× faster. Your CI pipeline will thank you.

## Behind the scenes

```
your files → gigatoken Rust core → token counts → your terminal
              (24 GB/s, baby)
```

Skips `__pycache__`, `.egg-info`, `.git`, `node_modules`, `.pytest_cache`, and friends by default. Because nobody needs to know how many tokens their build artifacts are.

Binary files are skipped with a single warning. Empty files return 0. Hidden files are ignored unless you `--hidden`. Basically it does what you'd expect a token counter to do.

## License

MIT
