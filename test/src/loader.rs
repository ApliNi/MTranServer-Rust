use std::fs;
use std::path::{Path, PathBuf};
use serde::de::DeserializeOwned;

/// Returns the path to the `cases` directory located inside this test crate.
pub fn cases_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("cases")
}

/// Resolves a relative path inside the `cases` directory.
pub fn resolve_case_path(rel_path: impl AsRef<Path>) -> PathBuf {
    cases_dir().join(rel_path)
}

/// Reads and parses a JSON test case fixture.
pub fn load_json<T: DeserializeOwned>(rel_path: impl AsRef<Path>) -> T {
    let path = resolve_case_path(rel_path.as_ref());
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read JSON test case from {:?}: {}", path, err));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse JSON test case from {:?}: {}", path, err))
}

/// Reads and parses a TOML test case fixture.
pub fn load_toml<T: DeserializeOwned>(rel_path: impl AsRef<Path>) -> T {
    let path = resolve_case_path(rel_path.as_ref());
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read TOML test case from {:?}: {}", path, err));
    toml::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse TOML test case from {:?}: {}", path, err))
}

/// Reads a plain text test case fixture (such as Markdown or raw text).
pub fn load_text(rel_path: impl AsRef<Path>) -> String {
    let path = resolve_case_path(rel_path.as_ref());
    fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read text test fixture from {:?}: {}", path, err))
}
