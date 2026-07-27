# tokn

A CLI to easily calculate how many tokens your files take. Because context pollution exists and you need to know *exactly* how much of your precious 200K window that 15,000-line YAML config is about to vaporize.

Powered by [gigatoken](https://github.com/marcelroed/gigatoken) under the hood, because `tiktoken` was built for prompt-sized strings and this thing runs at **24 GB/s** for when you're counting *everything*.

## Install

tokn is a native Rust binary — no Python, no pip, no venv.

### Pre-built binaries (coming soon)

```bash
# macOS / Linux
curl -fsSL https://github.com/HenryGetz/tokn/releases/latest/download/tokn-$(uname -s)-$(uname -m) -o tokn
chmod +x tokn
sudo mv tokn /usr/local/bin/
```

### From source

```bash
git clone https://github.com/HenryGetz/tokn.git
cd tokn
cargo +nightly build -Zprofile-rustflags --release
sudo cp target/release/tokn /usr/local/bin/
```

Requires [Rust](https://rustup.rs) nightly toolchain.

### Shell completions

```bash
tokn completions bash > ~/.local/share/bash-completion/completions/tokn
tokn completions zsh  > ~/.zfunc/_tokn
tokn completions fish > ~/.config/fish/completions/tokn.fish
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
$ tokn --per-file --sort
        2,759  src/main.rs
          115  Cargo.toml
           14  .gitignore
──────────
        2,888
```

JSON for the spreadsheet people:

```bash
$ tokn --json
{
  "files": [
    {"path": "src/main.rs", "tokens": 2759, "skipped": false}
  ],
  "total": 2759
}
```

Limit directory depth:

```bash
$ tokn --max-depth 1
1,234
```

Verbose mode (see what's happening):

```bash
$ tokn --verbose
Info: using tokenizer: Xenova/gpt-4o
Info: found 47 files
Info: counted 47 files, skipped 0 unreadable
3,756
```

## Flags

| Flag | What it does |
|------|-------------|
| `--ext .py` `-e .py` | Only count files with this extension. Repeatable. |
| `--exclude-ext .pyc` `-E .pyc` | Skip files with this extension. Repeatable. |
| `--hidden` `-H` | Count hidden files too. Hope you know what you're doing. |
| `--model gpt-4o` `-m gpt-4o` | Model/tokenizer to use. Different models, different counts. |
| `--per-file` `-p` | Show per-file breakdown. |
| `--json` `-j` | Output as JSON. |
| `--sort` `-s` | Sort per-file output by token count (highest first). |
| `--sort-reverse` `-S` | Sort per-file output by token count (lowest first). |
| `--max-depth N` `-d N` | Limit directory recursion depth (0 = no subdirs). |
| `--no-ignore` `-I` | Don't skip `__pycache__`, `.git`, `node_modules`, etc. |
| `--verbose` `-v` | Print diagnostic info to stderr. |
| `--quiet` `-q` | Suppress all warnings. |
| `--color auto` | Terminal color control: `always`, `never`, or `auto` (default). |
| `--version` `-V` | Show version info. |
| `tokn completions SHELL` | Generate shell completions. |

## Why not just use `wc -c`?

Because `wc -c` counts bytes, not tokens. And those are *very* different numbers. Go ahead, paste 100KB of Python into ChatGPT and tell me how many tokens that was. I'll wait.

## Why not just use `tiktoken`?

You totally can. `tiktoken` is great for `len(enc.encode("hello world"))`. `gigatoken` (what powers this) is a Rust BPE tokenizer that hits **~24 GB/s** on large files. That's roughly 680× faster. Your CI pipeline will thank you.

## Behind the scenes

```
your files → gigatoken Rust core → token counts → your terminal
              (24 GB/s, baby)
```

tokn itself is pure Rust — the entire heavy-lift tokenization pipeline runs in native code with no Python FFI overhead. First run downloads the tokenizer from HuggingFace Hub (~1 MB) and caches it to `~/.cache/huggingface/`. Subsequent runs are instant.

Skips `__pycache__`, `.egg-info`, `.git`, `node_modules`, `.pytest_cache`, and friends by default. Because nobody needs to know how many tokens their build artifacts are.

Binary files are skipped with a single warning. Empty files return 0. Hidden files are ignored unless you `--hidden`. Basically it does what you'd expect a token counter to do.

## License

MIT
