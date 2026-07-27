//! Integration tests for the tokn CLI.
//!
//! These tests use `assert_cmd` and `tempfile` to run the actual binary
//! and verify its behavior against real files on disk.
//!
//! All tokenizing tests use `--model openai-community/gpt2`, which is
//! universally available and small. The first run downloads the tokenizer
//! from HuggingFace Hub (~1 MB); subsequent runs use the cache at
//! `~/.cache/huggingface/`.
//!
//! **NOTE**: `TempDir` creates dot-prefixed dirs on Linux (e.g. `.tmpXXXX`),
//! which the CLI treats as hidden if `--hidden` is not given. Most tests
//! add `--hidden` to handle this. The `test_hidden_files_excluded` test
//! uses a non-hidden subdirectory instead.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// Shorthand for a command pre-configured with the model and color disabled.
fn tokn_cmd() -> Command {
    let mut cmd = Command::cargo_bin("tokn").unwrap();
    cmd.arg("--model").arg("openai-community/gpt2");
    cmd.arg("--color").arg("never");
    cmd
}

/// Shorthand that also adds `--hidden`.
fn tokn_cmd_hidden() -> Command {
    let mut cmd = tokn_cmd();
    cmd.arg("--hidden");
    cmd
}

/// Write a UTF-8 file into `dir` with the given relative path and content.
fn write_file(dir: &Path, relative_path: &str, content: &str) {
    let full = dir.join(relative_path);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

// ---------------------------------------------------------------------------
// 1. test_simple_count
// ---------------------------------------------------------------------------

#[test]
fn test_simple_count() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    let assert = tokn_cmd()
        .arg(tmp.path())
        .assert()
        .success();

    let output = std::str::from_utf8(&assert.get_output().stdout)
        .unwrap()
        .trim()
        .to_string();
    let count: usize = output
        .parse()
        .expect(&format!("Expected numeric token count, got: {:?}", output));
    assert!(count > 0, "Expected non-zero token count, got: {}", count);
}

// ---------------------------------------------------------------------------
// 2. test_json_output
// ---------------------------------------------------------------------------

#[test]
fn test_json_output() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    let assert = tokn_cmd()
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(stdout).expect("Expected valid JSON output");

    assert!(parsed["files"].is_array(), "Missing 'files' array");
    assert!(parsed["total"].is_number(), "Missing 'total' field");

    let files = parsed["files"].as_array().unwrap();
    assert!(!files.is_empty(), "Expected at least one file entry");

    let first_path = files[0]["path"].as_str().unwrap();
    assert!(
        first_path.contains("hello.txt"),
        "Expected 'hello.txt' in path, got: {}",
        first_path
    );
}

// ---------------------------------------------------------------------------
// 3. test_per_file_output
// ---------------------------------------------------------------------------

#[test]
fn test_per_file_output() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.txt", "hello");
    write_file(tmp.path(), "b.txt", "world");

    let assert = tokn_cmd()
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();

    assert!(stdout.contains("a.txt"), "Output should contain 'a.txt'");
    assert!(stdout.contains("b.txt"), "Output should contain 'b.txt'");

    // Should have a separator line (Unicode box-drawing)
    assert!(
        stdout.contains('─'),
        "Output should contain a separator line (box drawing characters)"
    );
}

// ---------------------------------------------------------------------------
// 4. test_extension_filter
// ---------------------------------------------------------------------------

#[test]
fn test_extension_filter() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.py", "print('hello')");
    write_file(tmp.path(), "b.rs", "fn main() {}");
    write_file(tmp.path(), "c.txt", "hello world");

    let assert = tokn_cmd()
        .arg("--ext")
        .arg("py")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("a.py"), "Expected 'a.py' in output");
    assert!(!stdout.contains("b.rs"), "b.rs should NOT appear (not .py)");
    assert!(!stdout.contains("c.txt"), "c.txt should NOT appear (not .py)");
}

// ---------------------------------------------------------------------------
// 5. test_no_extension_files_pass
// ---------------------------------------------------------------------------

#[test]
fn test_no_extension_files_pass() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "Makefile", "all:\n\techo hello");
    write_file(tmp.path(), "a.py", "print('hello')");

    let assert = tokn_cmd()
        .arg("--ext")
        .arg("py")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(
        stdout.contains("Makefile"),
        "Expected 'Makefile' in output (no-extension files should pass through)"
    );
    assert!(stdout.contains("a.py"), "Expected 'a.py' in output");
}

// ---------------------------------------------------------------------------
// 6. test_exclude_extension
// ---------------------------------------------------------------------------

#[test]
fn test_exclude_extension() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.py", "print('hello')");
    write_file(tmp.path(), "b.pyc", "compiled bytecode");

    let assert = tokn_cmd()
        .arg("--exclude-ext")
        .arg("pyc")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("a.py"), "Expected 'a.py' in output");
    assert!(!stdout.contains("b.pyc"), "b.pyc should NOT appear (excluded)");
}

// ---------------------------------------------------------------------------
// 7. test_hidden_files_excluded
// ---------------------------------------------------------------------------
// NOTE: TempDir creates hidden directories on Linux, so we can't use it
// for testing hidden file exclusion without --hidden. Instead, we create
// a non-hidden subdirectory inside the temp dir.

#[test]
fn test_hidden_files_excluded() {
    let tmp = TempDir::new().unwrap();
    // Create a non-hidden work directory inside the hidden temp dir
    let work_dir = tmp.path().join("work");
    fs::create_dir(&work_dir).unwrap();
    write_file(&work_dir, ".hidden", "secret");
    write_file(&work_dir, "visible", "public");

    // Without --hidden, .hidden should not appear
    let assert = tokn_cmd()
        .arg("--per-file")
        .arg(&work_dir)
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(
        !stdout.contains(".hidden"),
        "'.hidden' should NOT appear without --hidden"
    );
    assert!(stdout.contains("visible"), "'visible' should appear");

    // With --hidden, .hidden should appear
    let assert2 = tokn_cmd_hidden()
        .arg("--per-file")
        .arg(&work_dir)
        .assert()
        .success();

    let stdout2 = std::str::from_utf8(&assert2.get_output().stdout).unwrap();
    assert!(
        stdout2.contains(".hidden"),
        "'.hidden' SHOULD appear with --hidden"
    );
    assert!(stdout2.contains("visible"), "'visible' should appear");
}

// ---------------------------------------------------------------------------
// 8. test_binary_file_skip
// ---------------------------------------------------------------------------

#[test]
fn test_binary_file_skip() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "valid.txt", "hello world");
    // Write raw non-UTF-8 bytes
    fs::write(tmp.path().join("binary.bin"), &[0xFF, 0xFE, 0x00, 0x00]).unwrap();

    let assert = tokn_cmd()
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();
    let files = parsed["files"].as_array().unwrap();

    let valid_entry = files
        .iter()
        .find(|f| f["path"].as_str().unwrap().contains("valid.txt"))
        .unwrap();
    // skipped field is always present in JSON output (true = binary/unreadable, false = normal)
    let valid_skipped = valid_entry["skipped"].as_bool().unwrap_or(false);
    assert!(!valid_skipped, "valid.txt should not be skipped");

    let binary_entry = files
        .iter()
        .find(|f| f["path"].as_str().unwrap().contains("binary.bin"))
        .unwrap();
    let binary_skipped = binary_entry["skipped"].as_bool().unwrap_or(false);
    assert!(binary_skipped, "binary.bin should be skipped");
}

// ---------------------------------------------------------------------------
// 9. test_empty_file
// ---------------------------------------------------------------------------

#[test]
fn test_empty_file() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "empty.txt", "");

    let assert = tokn_cmd()
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();

    assert_eq!(parsed["total"].as_u64().unwrap(), 0);
    let files = parsed["files"].as_array().unwrap();
    assert_eq!(files[0]["tokens"].as_u64().unwrap(), 0);
    // skipped field is always present in JSON output (true = binary/unreadable, false = normal)
    let skipped = files[0]["skipped"].as_bool().unwrap_or(false);
    assert!(!skipped, "Empty file should NOT be skipped");
}

// ---------------------------------------------------------------------------
// 10. test_stdin_input
// ---------------------------------------------------------------------------

#[test]
fn test_stdin_input() {
    // NOTE: stdin mode is triggered when no paths given and stdin is piped.
    // We write_stdin here which assert_cmd handles via a pipe.
    let assert = tokn_cmd()
        .write_stdin("hello world")
        .assert()
        .success();

    let output = std::str::from_utf8(&assert.get_output().stdout)
        .unwrap()
        .trim()
        .to_string();
    let count: usize = output.parse().unwrap();
    assert!(
        count > 0,
        "Expected non-zero token count from stdin, got: {}",
        count
    );
}

// ---------------------------------------------------------------------------
// 11. test_subdirectory_traversal
// ---------------------------------------------------------------------------

#[test]
fn test_subdirectory_traversal() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.py", "x = 1");
    write_file(tmp.path(), "sub/b.py", "y = 2");
    write_file(tmp.path(), "sub/deep/c.py", "z = 3");

    let assert = tokn_cmd()
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();
    let files = parsed["files"].as_array().unwrap();

    assert_eq!(files.len(), 3, "Expected 3 files from recursive traversal");

    let paths: Vec<&str> = files.iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert!(paths.iter().any(|p| p.contains("a.py")), "Missing a.py");
    assert!(paths.iter().any(|p| p.contains("b.py")), "Missing b.py");
    assert!(paths.iter().any(|p| p.contains("c.py")), "Missing c.py");
}

// ---------------------------------------------------------------------------
// 12. test_skip_dirs
// ---------------------------------------------------------------------------

#[test]
fn test_skip_dirs() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "__pycache__/cached.py", "# cached");
    write_file(tmp.path(), ".git/config", "[core]");
    write_file(tmp.path(), "node_modules/pkg/index.js", "module.exports = {};");
    write_file(tmp.path(), "main.py", "print('hello')");

    let assert = tokn_cmd()
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();

    // Only main.py should appear
    assert!(stdout.contains("main.py"), "Expected 'main.py' in output");
    assert!(
        !stdout.contains("__pycache__"),
        "__pycache__ should not appear"
    );
    assert!(!stdout.contains(".git"), ".git should not appear");
    assert!(
        !stdout.contains("node_modules"),
        "node_modules should not appear"
    );
}

// ---------------------------------------------------------------------------
// 13. test_version_flag
// ---------------------------------------------------------------------------

#[test]
fn test_version_flag() {
    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--version")
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("tokn"), "Version output should contain 'tokn'");
    assert!(
        stdout.contains('0'),
        "Version output should contain a version number, got: {}",
        stdout.trim()
    );
}

// ---------------------------------------------------------------------------
// 14. test_no_files
// ---------------------------------------------------------------------------

#[test]
fn test_no_files() {
    let tmp = TempDir::new().unwrap();
    // tmp directory is empty by default

    let assert = tokn_cmd()
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .code(2); // exit code 2 = no files found

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();

    assert_eq!(parsed["total"].as_u64().unwrap(), 0);
    let files = parsed["files"].as_array().unwrap();
    assert!(
        files.is_empty(),
        "Expected empty files array for empty dir"
    );
}

// ---------------------------------------------------------------------------
// 15. test_version_flag_long
// ---------------------------------------------------------------------------

#[test]
fn test_version_flag_long() {
    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("-V")
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("tokn"), "Version output should contain 'tokn'");
    assert!(
        stdout.contains("gigatoken"),
        "Version output should contain 'gigatoken'"
    );
}

// ---------------------------------------------------------------------------
// 16. test_single_file_argument
// ---------------------------------------------------------------------------

#[test]
fn test_single_file_argument() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "foo.py", "print('hello')");

    let assert = tokn_cmd()
        .arg("--per-file")
        .arg(tmp.path().join("foo.py"))
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("foo.py"), "Should contain foo.py when passed as single file");
}

// ---------------------------------------------------------------------------
// 17. test_multiple_extensions
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_extensions() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.py", "x=1");
    write_file(tmp.path(), "b.rs", "fn main(){}");
    write_file(tmp.path(), "c.txt", "hello");

    let assert = tokn_cmd()
        .arg("--ext").arg("py")
        .arg("--ext").arg("rs")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("a.py"), "a.py should appear (py extension)");
    assert!(stdout.contains("b.rs"), "b.rs should appear (rs extension)");
    assert!(!stdout.contains("c.txt"), "c.txt should NOT appear (not py or rs)");
}

// ---------------------------------------------------------------------------
// 18. test_combined_ext_and_exclude
// ---------------------------------------------------------------------------

#[test]
fn test_combined_ext_and_exclude() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.py", "x=1");
    write_file(tmp.path(), "b.pyc", "compiled");
    write_file(tmp.path(), "c.rs", "fn main(){}");

    let assert = tokn_cmd()
        .arg("--ext").arg("py")
        .arg("--ext").arg("pyc")
        .arg("--exclude-ext").arg("pyc")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("a.py"), "a.py should appear (py in include, not in exclude)");
    assert!(!stdout.contains("b.pyc"), "b.pyc should NOT appear (in both include and exclude, exclude wins)");
    assert!(!stdout.contains("c.rs"), "c.rs should NOT appear (rs not in include)");
}

// ---------------------------------------------------------------------------
// 19. test_extension_case_insensitive
// ---------------------------------------------------------------------------

#[test]
fn test_extension_case_insensitive() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "a.PY", "x=1");
    write_file(tmp.path(), "b.RS", "fn main(){}");

    let assert = tokn_cmd()
        .arg("--ext").arg("py")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("a.PY"), "a.PY should appear (case-insensitive extension match)");
    assert!(!stdout.contains("b.RS"), "b.RS should NOT appear (rs != py)");
}

// ---------------------------------------------------------------------------
// 20. test_env_var_default_model
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_default_model() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--color").arg("never")
        .env("TOKN_MODEL", "openai-community/gpt2")
        .arg(tmp.path())
        .assert()
        .success();

    let output = std::str::from_utf8(&assert.get_output().stdout).unwrap().trim().to_string();
    let count: usize = output.parse().unwrap();
    assert!(count > 0, "Env var should set default model");
}

// ---------------------------------------------------------------------------
// 21. test_env_var_overridden_by_flag
// ---------------------------------------------------------------------------

#[test]
fn test_env_var_overridden_by_flag() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    // Set TOKN_MODEL to something that would fail, but override with --model
    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--color").arg("never")
        .arg("--model").arg("openai-community/gpt2")
        .env("TOKN_MODEL", "nonexistent-model-999")
        .arg(tmp.path())
        .assert()
        .success();

    let output = std::str::from_utf8(&assert.get_output().stdout).unwrap().trim().to_string();
    let count: usize = output.parse().unwrap();
    assert!(count > 0, "Explicit --model should override TOKN_MODEL env var");
}

// ---------------------------------------------------------------------------
// 22. test_sort_order
// ---------------------------------------------------------------------------

#[test]
fn test_sort_order() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "small.txt", "x");
    write_file(tmp.path(), "medium.txt", "hello world");
    // ~2 tokens per word, "hello world" repeated 10 times
    write_file(tmp.path(), "large.txt", &"hello world ".repeat(10));

    let assert = tokn_cmd()
        .arg("--sort")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    // small.txt should appear before large.txt in sort (small count on top)
    // Since --sort is descending by default, large.txt should appear first
    let large_pos = stdout.find("large.txt").unwrap();
    let small_pos = stdout.find("small.txt").unwrap();
    let medium_pos = stdout.find("medium.txt").unwrap();
    assert!(large_pos < medium_pos, "large.txt should appear before medium.txt");
    assert!(medium_pos < small_pos, "medium.txt should appear before small.txt");
}

// ---------------------------------------------------------------------------
// 23. test_sort_reverse_order
// ---------------------------------------------------------------------------

#[test]
fn test_sort_reverse_order() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "small.txt", "x");
    write_file(tmp.path(), "medium.txt", "hello world");
    write_file(tmp.path(), "large.txt", &"hello world ".repeat(10));

    let assert = tokn_cmd()
        .arg("--sort-reverse")
        .arg("--per-file")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let large_pos = stdout.find("large.txt").unwrap();
    let small_pos = stdout.find("small.txt").unwrap();
    let medium_pos = stdout.find("medium.txt").unwrap();
    assert!(small_pos < medium_pos, "small.txt should appear before medium.txt (reverse)");
    assert!(medium_pos < large_pos, "medium.txt should appear before large.txt (reverse)");
}

// ---------------------------------------------------------------------------
// 24. test_verbose_output
// ---------------------------------------------------------------------------

#[test]
fn test_verbose_output() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    let assert = tokn_cmd()
        .arg("--verbose")
        .arg(tmp.path())
        .assert()
        .success();

    let stderr = std::str::from_utf8(&assert.get_output().stderr).unwrap();
    assert!(stderr.contains("Info:"), "Verbose stderr should contain 'Info:'");
    assert!(stderr.contains("tokenizer"), "Verbose stderr should mention tokenizer");
    assert!(stderr.contains("found"), "Verbose stderr should mention file count");
}

// ---------------------------------------------------------------------------
// 25. test_quiet_suppresses_warnings
// ---------------------------------------------------------------------------

#[test]
fn test_quiet_suppresses_warnings() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "valid.txt", "hello");
    fs::write(tmp.path().join("binary.bin"), &[0xFF, 0xFE]).unwrap();

    let assert = tokn_cmd()
        .arg("--quiet")
        .arg(tmp.path())
        .assert()
        .success();

    let stderr = std::str::from_utf8(&assert.get_output().stderr).unwrap();
    assert!(!stderr.contains("Warning:"), "Quiet mode should suppress warnings");
}

// ---------------------------------------------------------------------------
// 26. test_max_depth_cli
// ---------------------------------------------------------------------------

#[test]
fn test_max_depth_cli() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "root.txt", "x");
    write_file(tmp.path(), "sub/sub.txt", "hello world");
    write_file(tmp.path(), "sub/deep/deep.txt", "buried");

    let assert = tokn_cmd()
        .arg("--max-depth").arg("1")
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();
    let files = parsed["files"].as_array().unwrap();

    assert!(files.iter().any(|f| f["path"].as_str().unwrap().contains("root.txt")), "root.txt should appear");
    assert!(files.iter().any(|f| f["path"].as_str().unwrap().contains("sub.txt")), "sub.txt should appear at depth 1");
    assert!(!files.iter().any(|f| f["path"].as_str().unwrap().contains("deep.txt")), "deep.txt should NOT appear at max-depth 1");
}

// ---------------------------------------------------------------------------
// 27. test_color_always_output
// ---------------------------------------------------------------------------

#[test]
fn test_color_always_output() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--color").arg("always")
        .arg("--model").arg("openai-community/gpt2")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("\x1b["), "Color always should produce ANSI codes");
}

// ---------------------------------------------------------------------------
// 28. test_color_never_output
// ---------------------------------------------------------------------------

#[test]
fn test_color_never_output() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "hello.txt", "hello world");

    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--color").arg("never")
        .arg("--model").arg("openai-community/gpt2")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(!stdout.contains("\x1b["), "Color never should NOT produce ANSI codes");
}

// ---------------------------------------------------------------------------
// 29. test_no_ignore_flag
// ---------------------------------------------------------------------------

#[test]
fn test_no_ignore_flag() {
    let tmp = TempDir::new().unwrap();
    write_file(tmp.path(), "main.py", "print('hello')");
    write_file(tmp.path(), "__pycache__/cached.pyc", "compiled");

    let assert = tokn_cmd()
        .arg("--no-ignore")
        .arg("--json")
        .arg(tmp.path())
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();
    let files = parsed["files"].as_array().unwrap();

    assert!(files.iter().any(|f| f["path"].as_str().unwrap().contains("main.py")));
    assert!(files.iter().any(|f| f["path"].as_str().unwrap().contains("cached.pyc")), "cached.pyc should appear with --no-ignore");
}

// ---------------------------------------------------------------------------
// 30. test_completions_bash
// ---------------------------------------------------------------------------

#[test]
fn test_completions_bash() {
    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("completions")
        .arg("bash")
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("complete"), "Bash completion should contain 'complete'");
}

// ---------------------------------------------------------------------------
// 31. test_exit_code_error
// ---------------------------------------------------------------------------

#[test]
fn test_exit_code_error() {
    Command::cargo_bin("tokn")
        .unwrap()
        .arg("--color").arg("never")
        .arg("--model").arg("definitely-not-a-real-model-xyz-12345")
        .arg(".")
        .assert()
        .code(1); // error exit code
}

// ---------------------------------------------------------------------------
// 32. test_help_contains_examples
// ---------------------------------------------------------------------------

#[test]
fn test_help_contains_examples() {
    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--help")
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    assert!(stdout.contains("Examples:"), "Help should contain Examples section");
}

// ---------------------------------------------------------------------------
// 33. test_help_contains_all_flags
// ---------------------------------------------------------------------------

#[test]
fn test_help_contains_all_flags() {
    let assert = Command::cargo_bin("tokn")
        .unwrap()
        .arg("--help")
        .assert()
        .success();

    let stdout = std::str::from_utf8(&assert.get_output().stdout).unwrap();
    for flag in &["--ext", "--exclude-ext", "--hidden", "--model", "--per-file",
                   "--json", "--sort", "--sort-reverse", "--max-depth", "--verbose", "--quiet",
                   "--color", "--no-ignore"] {
        assert!(stdout.contains(flag), "Help should contain flag: {}", flag);
    }
}
