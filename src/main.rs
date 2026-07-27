use clap::{Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use eyre::{Context, Result};
use owo_colors::OwoColorize;
use serde::Serialize;
use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Count GPT tokens in files, directories, or stdin.
///
/// tokn uses gigatoken under the hood — a SIMD-optimized Rust tokenizer
/// that hits ~24 GB/s on large files. Perfect for figuring out how much
/// of your context window that 15,000-line YAML config is about to eat.
#[derive(Parser)]
#[command(
    name = "tokn",
    version,
    about,
    after_long_help = "Examples:\n  \
        # Count tokens in the current directory\n  \
        tokn\n\n  \
        # Count with per-file breakdown, sorted by size\n  \
        tokn --per-file --sort\n\n  \
        # Count only Python and Rust files, excluding test files\n  \
        tokn --ext py --ext rs --exclude-ext pyc\n\n  \
        # Count a single file\n  \
        tokn src/main.rs\n\n  \
        # Pipe text from stdin\n  \
        echo \"hello world\" | tokn\n\n  \
        # Output as JSON for scripting\n  \
        tokn --json > counts.json\n\n  \
        # Limit directory recursion depth\n  \
        tokn --max-depth 2\n\n  \
        # Include hidden files (dotfiles)\n  \
        tokn --hidden\n\n  \
        # Generate shell completions\n  \
        tokn completions bash > ~/.local/share/bash-completion/completions/tokn"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// File or directory paths to process.
    ///
    /// Defaults to the current directory, or reads stdin if piped.
    #[arg(value_name = "PATH", value_hint = clap::ValueHint::AnyPath)]
    paths: Vec<String>,

    /// Include ONLY files with these extensions (repeatable).
    ///
    /// Without this flag, ALL file types are counted regardless of extension.
    /// Files without an extension (Makefile, Dockerfile, etc.) always pass through.
    #[arg(short = 'e', long = "ext", value_name = "EXT")]
    extensions: Vec<String>,

    /// Exclude files with these extensions (repeatable).
    #[arg(short = 'E', long = "exclude-ext", value_name = "EXT")]
    exclude_extensions: Vec<String>,

    /// Include hidden files and directories (dotfiles).
    #[arg(short = 'H', long = "hidden")]
    hidden: bool,

    /// Model name or HuggingFace repo ID for tokenizer resolution.
    ///
    /// Examples: gpt-4o, gpt-4, gpt-3.5-turbo, Xenova/gpt-4o,
    /// openai-community/gpt2, or a local path to tokenizer.json.
    ///
    /// Defaults to gpt-4o. Set TOKN_MODEL env var to change the default.
    #[arg(short = 'm', long = "model", value_name = "MODEL")]
    model: Option<String>,

    /// Show per-file token counts before the total.
    #[arg(short = 'p', long = "per-file")]
    per_file: bool,

    /// Output results as JSON.
    #[arg(short = 'j', long = "json")]
    json: bool,

    /// Sort per-file output by token count, highest first.
    ///
    /// Only meaningful with --per-file.
    #[arg(short = 's', long = "sort")]
    sort: bool,

    /// Sort per-file output by token count, lowest first.
    ///
    /// Implies --sort.
    #[arg(short = 'S', long = "sort-reverse")]
    sort_reverse: bool,

    /// Maximum directory recursion depth.
    ///
    /// 0 = only the given paths, 1 = one level deep, etc.
    /// No limit when omitted.
    #[arg(short = 'd', long = "max-depth", value_name = "DEPTH")]
    max_depth: Option<usize>,

    /// Terminal color output control.
    #[arg(long = "color", default_value = "auto", value_name = "WHEN")]
    color: ColorWhen,

    /// Print diagnostic info to stderr (model loaded, file count, skips).
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    /// Don't skip build artifact directories.
    ///
    /// By default, __pycache__, .git, node_modules, .venv, etc. are skipped.
    /// With this flag, every file is counted.
    #[arg(short = 'I', long = "no-ignore")]
    no_ignore: bool,

    /// Suppress all non-error stderr output (warnings).
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate shell completion script.
    ///
    /// Prints the completion script to stdout. Source it or save to
    /// your shell's completion directory.
    ///
    /// Examples:
    ///   tokn completions bash > ~/.local/share/bash-completion/completions/tokn
    ///   tokn completions zsh  > ~/.zfunc/_tokn
    ///   tokn completions fish > ~/.config/fish/completions/tokn.fish
    Completions {
        /// Shell to generate completions for.
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ColorWhen {
    Always,
    Never,
    Auto,
}

// ---------------------------------------------------------------------------
// Tokenizer backend
// ---------------------------------------------------------------------------

/// The gigatoken tokenizer loaded from a HuggingFace tokenizer.json.
type Tokenizer = gigatoken_rs::load_tokenizer::hf::HfTokenizer;

/// Count tokens in `text` using the given tokenizer.
fn count_tokens(tok: &mut Tokenizer, text: &str) -> usize {
    match tok {
        Tokenizer::Bpe(tok) => {
            let mut out: Vec<u32> = Vec::new();
            tok.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
            out.len()
        }
        Tokenizer::SentencePiece(tok) => tok.encode_raw(text).len(),
    }
}

// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------

/// Map short model names to HF repo IDs.
const MODEL_FALLBACK: &[(&str, &str)] = &[
    ("gpt-4o-mini", "Xenova/gpt-4o"),
    ("gpt-4o", "Xenova/gpt-4o"),
    ("gpt-4-turbo", "Xenova/gpt-4"),
    ("gpt-4-32k", "Xenova/gpt-4"),
    ("gpt-4", "Xenova/gpt-4"),
    ("gpt-3.5-turbo-16k", "Xenova/gpt-3.5-turbo"),
    ("gpt-3.5-turbo", "Xenova/gpt-3.5-turbo"),
];
const UNIVERSAL_FALLBACK: &str = "openai-community/gpt2";

/// Given a model specifier, load the tokenizer.
/// Returns (tokenizer, resolved_repo_name) for verbose output.
fn load_tokenizer(model: &str) -> Result<(Tokenizer, String)> {
    // 1. Is it a local .json file?
    let model_path = Path::new(model);
    if model_path.is_file() && model_path.extension().map_or(false, |ext| ext == "json") {
        let data = std::fs::read(model_path)
            .wrap_err_with(|| format!("Failed to read tokenizer file: {}", model))?;
        let tok = gigatoken_rs::load_tokenizer::hf::load_hf_slice(&data)
            .wrap_err("Failed to parse tokenizer.json")?;
        return Ok((tok, model.to_string()));
    }

    // 2. Determine the HF repo ID.
    // Check known short names FIRST — looks_like_repo_id would match
    // bare names like "gpt-4o" as valid repos, but those don't exist on HF.
    let repo_id: Cow<str> =
        if let Some(&(_, mapped)) = MODEL_FALLBACK.iter().find(|&&(k, _)| k == model) {
            Cow::Borrowed(mapped)
        } else if gigatoken_rs::load_tokenizer::hub::looks_like_repo_id(model) {
            Cow::Borrowed(model)
        } else {
            eprintln!(
                "{} unknown model '{}', falling back to {}",
                yellow_if_color("Warning:"),
                model,
                UNIVERSAL_FALLBACK
            );
            Cow::Borrowed(UNIVERSAL_FALLBACK)
        };

    // 3. Download / locate tokenizer.json from HF Hub.
    let cached_path =
        gigatoken_rs::load_tokenizer::hub::hub_file(&repo_id, "tokenizer.json", "main")
            .wrap_err_with(|| format!("Failed to download tokenizer for {}", repo_id))?;
    let data = std::fs::read(&cached_path)
        .wrap_err_with(|| format!("Failed to read cached tokenizer at {:?}", cached_path))?;
    let tok = gigatoken_rs::load_tokenizer::hf::load_hf_slice(&data)
        .wrap_err("Failed to parse tokenizer.json")?;
    Ok((tok, repo_id.into_owned()))
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

/// Directories that are always skipped (unless --no-ignore).
const SKIP_DIRS: &[&str] = &[
    "__pycache__",
    ".git",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
];

fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name) || name.ends_with(".egg-info")
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with('.') && s != ".")
        .unwrap_or(false)
}

/// Collect all regular files under the given root, respecting CLI filters.
fn collect_files(
    root: &Path,
    cli: &Cli,
    include_exts: &Option<HashSet<String>>,
    exclude_exts: &HashSet<String>,
) -> Vec<PathBuf> {
    let mut walker = WalkDir::new(root);
    if let Some(max_depth) = cli.max_depth {
        // walkdir depth 0 = root only, depth 1 = root + immediate children.
        // User-visible "max-depth 0" = no recursion = walkdir depth 1.
        walker = walker.max_depth(max_depth + 1);
    }
    let mut files = Vec::new();
    let skip_artifact_dirs = !cli.no_ignore;

    for entry in walker.into_iter().filter_entry(|e| {
        if e.file_type().is_dir() {
            let name = e.file_name().to_string_lossy();
            if skip_artifact_dirs && should_skip_dir(&name) {
                return false;
            }
            if e.path() != root && !cli.hidden && is_hidden(e) {
                return false;
            }
        }
        true
    }) {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                if !cli.quiet {
                    if let Some(path) = err.path() {
                        eprintln!(
                            "{} cannot access {}",
                            yellow_if_color("Warning:"),
                            path.display()
                        );
                    }
                }
                continue;
            }
        };

        if !entry.file_type().is_file() {
            continue;
        }

        if !cli.hidden && is_hidden(&entry) {
            continue;
        }

        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let dot_ext = dot_ext(ext);
            if let Some(ref inc) = include_exts {
                if !inc.contains(&dot_ext) {
                    continue;
                }
            }
            if exclude_exts.contains(&dot_ext) {
                continue;
            }
        }

        files.push(entry.into_path());
    }
    files.sort();
    files
}

fn normalize_extensions(exts: &[String]) -> Option<HashSet<String>> {
    if exts.is_empty() {
        None
    } else {
        Some(exts.iter().map(|e| dot_ext(e)).collect())
    }
}

fn normalize_exclude_exts(exts: &[String]) -> HashSet<String> {
    exts.iter().map(|e| dot_ext(e)).collect()
}

/// Returns `".ext"` formatted consistently for comparison.
fn dot_ext(ext: &str) -> String {
    format!(".{}", ext.trim_start_matches('.').to_lowercase())
}

// ---------------------------------------------------------------------------
// Stdin detection
// ---------------------------------------------------------------------------

fn stdin_is_pipe() -> bool {
    #[cfg(unix)]
    {
        unsafe {
            let mut statbuf: libc::stat = std::mem::zeroed();
            if libc::fstat(0, &mut statbuf) == 0 {
                let mode = statbuf.st_mode & libc::S_IFMT;
                mode == libc::S_IFIFO || mode == libc::S_IFREG
            } else {
                false
            }
        }
    }
    #[cfg(not(unix))]
    {
        !std::io::stdin().is_terminal()
    }
}

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct FileEntry {
    path: String,
    tokens: usize,
    skipped: bool,
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    files: Vec<FileEntry>,
    total: usize,
}

fn count_file(path: &Path, tok: &mut Tokenizer) -> Option<usize> {
    match std::fs::read_to_string(path) {
        Ok(text) => Some(count_tokens(tok, &text)),
        Err(_) => None,
    }
}

fn format_count(n: usize) -> String {
    let s = n.to_string();
    let len = s.len();
    if len <= 3 {
        return s;
    }
    let mut result = String::with_capacity(len + (len - 1) / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

fn normalize_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, Ordering};

static COLOR_ENABLED: AtomicBool = AtomicBool::new(true);

fn setup_color(when: &ColorWhen) {
    match when {
        ColorWhen::Always => {
            owo_colors::set_override(true);
            COLOR_ENABLED.store(true, Ordering::Relaxed);
        }
        ColorWhen::Never => {
            owo_colors::set_override(false);
            COLOR_ENABLED.store(false, Ordering::Relaxed);
        }
        ColorWhen::Auto => {
            COLOR_ENABLED.store(true, Ordering::Relaxed);
        }
    }
}

fn styled_count(n: usize) -> String {
    let s = format_count(n);
    if COLOR_ENABLED.load(Ordering::Relaxed) {
        if n > 0 {
            s.green().bold().to_string()
        } else {
            s.dimmed().to_string()
        }
    } else {
        s
    }
}

fn dim_if_color(s: String) -> String {
    if COLOR_ENABLED.load(Ordering::Relaxed) {
        s.dimmed().to_string()
    } else {
        s
    }
}

fn yellow_if_color(s: &str) -> String {
    if COLOR_ENABLED.load(Ordering::Relaxed) {
        s.yellow().to_string()
    } else {
        s.to_string()
    }
}

fn red_if_color(s: &str) -> String {
    if COLOR_ENABLED.load(Ordering::Relaxed) {
        s.red().bold().to_string()
    } else {
        s.to_string()
    }
}

fn green_bold_if_color(s: &str) -> String {
    if COLOR_ENABLED.load(Ordering::Relaxed) {
        s.green().bold().to_string()
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_version() {
    println!(
        "{} {} (gigatoken {})",
        green_bold_if_color("tokn"),
        dim_if_color(env!("CARGO_PKG_VERSION").to_string()),
        dim_if_color("0.10.0".to_string())
    );
}

fn print_results(
    entries: &[FileEntry],
    total: usize,
    cli: &Cli,
    skipped_count: usize,
) {
    // JSON output
    if cli.json {
        let output = JsonOutput {
            files: entries.to_vec(),
            total,
        };
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return;
    }

    // Per-file output
    if cli.per_file {
        // Sort if requested
        let mut display = entries.to_vec();
        if cli.sort || cli.sort_reverse {
            if cli.sort_reverse {
                display.sort_by_key(|e| e.tokens);
            } else {
                display.sort_by_key(|e| std::cmp::Reverse(e.tokens));
            }
        }

        for entry in &display {
            let count_str = if entry.skipped {
                dim_if_color("(skip)".to_string())
            } else {
                styled_count(entry.tokens)
            };
            println!("{:>10}  {}", count_str, dim_if_color(entry.path.clone()));
        }

        println!("{}", dim_if_color("──────────".to_string()));
        println!("{:>10}", styled_count(total));

        if cli.verbose && skipped_count > 0 {
            eprintln!(
                "{} {} files, {} skipped",
                    dim_if_color("Info:".to_string()),
                entries.len(),
                skipped_count
            );
        }
        return;
    }

    // Plain total
    println!("{}", styled_count(total));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RunStats {
    files_found: usize,
    files_counted: usize,
    files_skipped: usize,
    total_tokens: usize,
    resolver: String,
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    // Setup color
    setup_color(&cli.color);

    // Handle completions subcommand
    if let Some(Commands::Completions { shell }) = &cli.command {
        let mut cmd = <Cli as clap::CommandFactory>::command();
        let name = cmd.get_name().to_string();
        clap_complete::generate(*shell, &mut cmd, name, &mut io::stdout());
        return Ok(());
    }

    // Resolve model: CLI flag > TOKN_MODEL env var > hardcoded default
    let (model, model_source) = if let Some(ref m) = cli.model {
        (m.clone(), "from --model flag")
    } else if let Ok(env_model) = std::env::var("TOKN_MODEL") {
        let env_model = env_model.trim();
        if !env_model.is_empty() {
            (env_model.to_string(), "from TOKN_MODEL env")
        } else {
            ("gpt-4o".to_string(), "default")
        }
    } else {
        ("gpt-4o".to_string(), "default")
    };

    // Load tokenizer
    let (mut tokenizer, resolver) = load_tokenizer(&model)?;
    let mut stats = RunStats {
        resolver,
        ..Default::default()
    };

    if cli.verbose {
        eprintln!(
            "{} using tokenizer: {} ({})",
            dim_if_color("Info:".to_string()),
            stats.resolver.bold(),
            model_source
        );
    }

    // Determine paths
    let paths: Vec<String> = if cli.paths.is_empty() {
        if stdin_is_pipe() {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .wrap_err("Failed to read stdin")?;
            let count = count_tokens(&mut tokenizer, &buf);

            if cli.verbose {
                eprintln!("{} read {} tokens from stdin", dim_if_color("Info:".to_string()), count);
            }

            if cli.json {
                let output = JsonOutput {
                    files: vec![FileEntry {
                        path: "(stdin)".into(),
                        tokens: count,
                        skipped: false,
                    }],
                    total: count,
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("{}", styled_count(count));
            }
            return Ok(());
        }
        vec![".".into()]
    } else {
        cli.paths.clone()
    };

    // Normalize extension filters
    let include_exts = normalize_extensions(&cli.extensions);
    let exclude_exts = normalize_exclude_exts(&cli.exclude_extensions);

    // Collect files from all paths
    let mut all_files: Vec<PathBuf> = Vec::new();
    for raw_path in &paths {
        let path = Path::new(raw_path);
        if !path.exists() {
            if !cli.quiet {
                eprintln!(
                    "{} path does not exist: {}",
                    yellow_if_color("Warning:"),
                    raw_path
                );
            }
            continue;
        }
        if path.is_file() {
            let ext_passes = if let Some(ref inc) = include_exts {
                path.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| inc.contains(&dot_ext(e)))
                    .unwrap_or(true)
            } else {
                true
            };
            let ex_ext_passes = if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                !exclude_exts.contains(&dot_ext(ext))
            } else {
                true
            };
            let hidden_passes = cli.hidden
                || !path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with('.'))
                    .unwrap_or(false);

            if ext_passes && ex_ext_passes && hidden_passes {
                all_files.push(path.to_path_buf());
            }
        } else {
            let files = collect_files(path, &cli, &include_exts, &exclude_exts);
            all_files.extend(files);
        }
    }

    all_files.sort();
    all_files.dedup();
    stats.files_found = all_files.len();

    if all_files.is_empty() {
        if !cli.quiet {
            eprintln!("{} no files found", yellow_if_color("Warning:"));
        }
        if cli.json {
            let output = JsonOutput {
                files: vec![],
                total: 0,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("0");
        }
        std::process::exit(2);
    }

    if cli.verbose {
        eprintln!(
            "{} found {} file{}",
                dim_if_color("Info:".to_string()),
            stats.files_found,
            if stats.files_found == 1 { "" } else { "s" }
        );
    }

    // Count tokens
    let mut entries: Vec<FileEntry> = Vec::with_capacity(all_files.len());
    let mut total = 0usize;
    let mut warned_binary = false;

    for path in &all_files {
        match count_file(path, &mut tokenizer) {
            Some(count) => {
                total += count;
                stats.files_counted += 1;
                entries.push(FileEntry {
                    path: normalize_path(path),
                    tokens: count,
                    skipped: false,
                });
            }
            None => {
                if !cli.quiet && !warned_binary {
                    let example = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| path.to_string_lossy().to_string());
                    eprintln!(
                        "{} skipping binary/unreadable file (e.g. {})",
                        yellow_if_color("Warning:"),
                        example
                    );
                    warned_binary = true;
                }
                stats.files_skipped += 1;
                entries.push(FileEntry {
                    path: normalize_path(path),
                    tokens: 0,
                    skipped: true,
                });
            }
        }
    }

    stats.total_tokens = total;
    stats.files_skipped = all_files.len() - stats.files_counted;

    if cli.verbose && stats.files_skipped > 0 {
        eprintln!(
            "{} counted {} file{}, skipped {} unreadable",
                dim_if_color("Info:".to_string()),
            stats.files_counted,
            if stats.files_counted == 1 { "" } else { "s" },
            stats.files_skipped
        );
    }

    print_results(&entries, total, &cli, stats.files_skipped);

    Ok(())
}

fn main() {
    // Handle --version before clap for custom formatting
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        print_version();
        return;
    }

    match run() {
        Ok(()) => {}
        Err(err) => {
            eprintln!("{} {:?}", red_if_color("Error:"), err);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- dot_ext ------------------------------------------------------------

    #[test]
    fn test_dot_ext_with_dot() {
        assert_eq!(dot_ext(".py"), ".py");
        assert_eq!(dot_ext(".rs"), ".rs");
    }

    #[test]
    fn test_dot_ext_without_dot() {
        assert_eq!(dot_ext("py"), ".py");
        assert_eq!(dot_ext("rs"), ".rs");
    }

    #[test]
    fn test_dot_ext_uppercase() {
        assert_eq!(dot_ext(".RS"), ".rs");
        assert_eq!(dot_ext("RS"), ".rs");
        assert_eq!(dot_ext(".Py"), ".py");
        assert_eq!(dot_ext("PY"), ".py");
    }

    #[test]
    fn test_dot_ext_mixed_case() {
        assert_eq!(dot_ext(".Rs"), ".rs");
        assert_eq!(dot_ext("Py"), ".py");
    }

    // -- format_count -------------------------------------------------------

    #[test]
    fn test_format_count_zero() {
        assert_eq!(format_count(0), "0");
    }

    #[test]
    fn test_format_count_under_1000() {
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1), "1");
        assert_eq!(format_count(42), "42");
    }

    #[test]
    fn test_format_count_thousands() {
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1234), "1,234");
    }

    #[test]
    fn test_format_count_millions() {
        assert_eq!(format_count(1234567), "1,234,567");
        assert_eq!(format_count(1000000), "1,000,000");
    }

    // -- normalize_path -----------------------------------------------------

    #[test]
    fn test_normalize_path_unix() {
        let p = Path::new("foo/bar/baz.txt");
        assert_eq!(normalize_path(p), "foo/bar/baz.txt");
    }

    #[test]
    fn test_normalize_path_backslash_conversion() {
        let p = Path::new("foo\\bar\\baz.txt");
        assert_eq!(normalize_path(p), "foo/bar/baz.txt");
    }

    #[test]
    fn test_normalize_path_mixed_separators() {
        let p = Path::new("foo\\bar/baz.txt");
        assert_eq!(normalize_path(p), "foo/bar/baz.txt");
    }

    // -- normalize_extensions -----------------------------------------------

    #[test]
    fn test_normalize_extensions_empty() {
        let empty: &[String] = &[];
        assert_eq!(normalize_extensions(empty), None);
    }

    #[test]
    fn test_normalize_extensions_single() {
        let exts = vec!["py".to_string()];
        let result = normalize_extensions(&exts).unwrap();
        assert!(result.contains(".py"));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_normalize_extensions_multiple() {
        let exts = vec!["py".to_string(), ".rs".to_string(), "TS".to_string()];
        let result = normalize_extensions(&exts).unwrap();
        assert!(result.contains(".py"));
        assert!(result.contains(".rs"));
        assert!(result.contains(".ts"));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_normalize_extensions_all_dot_prefixed() {
        let exts = vec![".py".to_string(), ".rs".to_string()];
        let result = normalize_extensions(&exts).unwrap();
        assert!(result.contains(".py"));
        assert!(result.contains(".rs"));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_normalize_exclude_exts_basic() {
        let exts = vec!["pyc".to_string(), ".lock".to_string()];
        let result = normalize_exclude_exts(&exts);
        assert!(result.contains(".pyc"));
        assert!(result.contains(".lock"));
        assert_eq!(result.len(), 2);
    }

    // -- count_tokens -------------------------------------------------------

    #[test]
    fn test_count_tokens_basic() {
        let tokenizer = match load_tokenizer(UNIVERSAL_FALLBACK) {
            Ok((t, _)) => t,
            Err(_) => {
                panic!(
                    "Need network to download {} tokenizer on first run.\n\
                     Run with `cargo test -- --ignored` if offline, or set HF_HUB_CACHE.",
                    UNIVERSAL_FALLBACK
                );
            }
        };
        let mut tok = tokenizer;
        let count = count_tokens(&mut tok, "hello world");
        assert!(count > 0, "Expected >0 tokens for 'hello world'");
    }

    // -- should_skip_dir ----------------------------------------------------

    #[test]
    fn test_should_skip_dir_known() {
        assert!(should_skip_dir("__pycache__"));
        assert!(should_skip_dir(".git"));
        assert!(should_skip_dir("node_modules"));
        assert!(should_skip_dir(".pytest_cache"));
        assert!(should_skip_dir(".mypy_cache"));
        assert!(should_skip_dir(".tox"));
        assert!(should_skip_dir(".venv"));
        assert!(should_skip_dir("venv"));
    }

    #[test]
    fn test_should_skip_dir_egg_info() {
        assert!(should_skip_dir("myproject.egg-info"));
        assert!(should_skip_dir("tokn.egg-info"));
    }

    #[test]
    fn test_should_skip_dir_normal() {
        assert!(!should_skip_dir("src"));
        assert!(!should_skip_dir("tests"));
        assert!(!should_skip_dir("my_project"));
    }

    // -- is_hidden ----------------------------------------------------------

    #[test]
    fn test_is_hidden_dot_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".hidden"), "secret").unwrap();
        let entry = WalkDir::new(tmp.path())
            .into_iter()
            .find(|e| {
                e.as_ref()
                    .ok()
                    .map(|de| de.file_name() == ".hidden")
                    .unwrap_or(false)
            })
            .unwrap()
            .unwrap();
        assert!(is_hidden(&entry));
    }

    #[test]
    fn test_is_hidden_normal_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("visible"), "hello").unwrap();
        let entry = WalkDir::new(tmp.path())
            .into_iter()
            .find(|e| {
                e.as_ref()
                    .ok()
                    .map(|de| de.file_name() == "visible")
                    .unwrap_or(false)
            })
            .unwrap()
            .unwrap();
        assert!(!is_hidden(&entry));
    }

    // -- hidden root not pruned ---------------------------------------------

    #[test]
    fn test_hidden_root_not_pruned() {
        let tmp = tempfile::TempDir::new().unwrap();
        let hidden_root = tmp.path();
        std::fs::write(hidden_root.join("data.txt"), "hello world").unwrap();

        let cli = Cli {
            command: None,
            paths: vec![],
            extensions: vec![],
            exclude_extensions: vec![],
            hidden: false,
            model: Some("gpt-4o".into()),
            per_file: false,
            json: false,
            sort: false,
            sort_reverse: false,
            max_depth: None,
            color: ColorWhen::Auto,
            verbose: false,
            no_ignore: false,
            quiet: false,
        };
        let include_exts: Option<HashSet<String>> = None;
        let exclude_exts: HashSet<String> = HashSet::new();

        let files = collect_files(hidden_root, &cli, &include_exts, &exclude_exts);
        assert!(!files.is_empty(), "Hidden root directory should still be traversed");
        assert!(
            files.iter().any(|f| f.file_name().unwrap() == "data.txt"),
            "data.txt should be found in hidden root dir"
        );
    }

    // -- max_depth ----------------------------------------------------------

    #[test]
    fn test_max_depth_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("sub")).unwrap();
        std::fs::write(tmp.path().join("root.txt"), "x").unwrap();
        std::fs::write(tmp.path().join("sub/sub.txt"), "y").unwrap();

        let cli = Cli {
            command: None,
            paths: vec![],
            extensions: vec![],
            exclude_extensions: vec![],
            hidden: true,
            model: Some("gpt-4o".into()),
            per_file: false,
            json: false,
            sort: false,
            sort_reverse: false,
            max_depth: Some(0),
            color: ColorWhen::Auto,
            verbose: false,
            no_ignore: false,
            quiet: false,
        };
        let include_exts: Option<HashSet<String>> = None;
        let exclude_exts: HashSet<String> = HashSet::new();

        let files = collect_files(tmp.path(), &cli, &include_exts, &exclude_exts);
        // max_depth(0) => walkdir depth 1 = root + immediate children (no subdirs)
        assert_eq!(files.len(), 1, "max_depth=0 should only see files in root");
        assert!(files.iter().any(|f| f.file_name().unwrap() == "root.txt"));
    }

    #[test]
    fn test_max_depth_one() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("sub/deep")).unwrap();
        std::fs::write(tmp.path().join("root.txt"), "x").unwrap();
        std::fs::write(tmp.path().join("sub/sub.txt"), "y").unwrap();
        std::fs::write(tmp.path().join("sub/deep/deep.txt"), "z").unwrap();

        let cli = Cli {
            command: None,
            paths: vec![],
            extensions: vec![],
            exclude_extensions: vec![],
            hidden: true,
            model: Some("gpt-4o".into()),
            per_file: false,
            json: false,
            sort: false,
            sort_reverse: false,
            max_depth: Some(1),
            color: ColorWhen::Auto,
            verbose: false,
            no_ignore: false,
            quiet: false,
        };
        let include_exts: Option<HashSet<String>> = None;
        let exclude_exts: HashSet<String> = HashSet::new();

        let files = collect_files(tmp.path(), &cli, &include_exts, &exclude_exts);
        // max_depth(1) => walkdir depth 2 = root + children + grandchildren
        assert_eq!(files.len(), 2, "max_depth=1 should see root files + one level of subdirs");
        assert!(files.iter().any(|f| f.file_name().unwrap() == "root.txt"));
        assert!(
            files.iter().any(|f| f.file_name().unwrap() == "sub.txt"),
            "sub.txt should appear at max_depth=1"
        );
        assert!(
            !files.iter().any(|f| f.file_name().unwrap() == "deep.txt"),
            "max_depth=1 should NOT see files two levels deep"
        );
    }

    // -- no_ignore ----------------------------------------------------------

    #[test]
    fn test_no_ignore_includes_artifacts() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("__pycache__")).unwrap();
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        std::fs::write(tmp.path().join("main.py"), "x").unwrap();
        std::fs::write(tmp.path().join("__pycache__/cached.pyc"), "y").unwrap();
        std::fs::write(tmp.path().join(".git/config"), "[core]").unwrap();

        // Without --no-ignore
        let cli_default = Cli {
            command: None,
            paths: vec![],
            extensions: vec![],
            exclude_extensions: vec![],
            hidden: true,
            model: Some("gpt-4o".into()),
            per_file: false,
            json: false,
            sort: false,
            sort_reverse: false,
            max_depth: None,
            color: ColorWhen::Auto,
            verbose: false,
            no_ignore: false,
            quiet: false,
        };
        let none: Option<HashSet<String>> = None;
        let empty: HashSet<String> = HashSet::new();
        let files_default = collect_files(tmp.path(), &cli_default, &none, &empty);
        assert!(
            files_default.iter().any(|f| f.file_name().unwrap() == "main.py")
        );
        assert!(
            !files_default
                .iter()
                .any(|f| f.file_name().unwrap() == "cached.pyc")
        );
        assert!(
            !files_default
                .iter()
                .any(|f| f.file_name().unwrap() == "config")
        );

        // With --no-ignore
        let cli_noignore = Cli {
            no_ignore: true,
            ..cli_default
        };
        let files_noignore = collect_files(tmp.path(), &cli_noignore, &none, &empty);
        assert!(
            files_noignore
                .iter()
                .any(|f| f.file_name().unwrap() == "cached.pyc"),
            "cached.pyc should appear with --no-ignore"
        );
        assert!(
            files_noignore
                .iter()
                .any(|f| f.file_name().unwrap() == "config"),
            ".git/config should appear with --no-ignore"
        );
    }
}
