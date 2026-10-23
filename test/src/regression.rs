//! Translation quality regression runner.
//!
//! Re-translates test cases from `test/cases/regression.json` and updates
//! the `output` field in place so differences can be inspected via `git diff`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;

use serde::{Deserialize, Serialize};

use mtranserver_lib::config::{self, Config, ParseFormats};
use mtranserver_lib::{engine, models, service, utils};

static SERVICE_INIT: Once = Once::new();

/// A single translation regression test case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestCase {
    pub target: String,
    pub input: String,
    pub output: String,
}

/// Returns the default path to corpus: `test/cases/regression.json`.
pub fn default_corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("cases")
        .join("regression.json")
}

/// Loads regression test cases from a JSON file.
pub fn load_cases(path: impl AsRef<Path>) -> Result<Vec<TestCase>, String> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read regression cases from {:?}: {}", path, e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse regression cases JSON {:?}: {}", path, e))
}

/// Saves regression test cases to a JSON file.
pub fn save_cases(path: impl AsRef<Path>, cases: &[TestCase]) -> Result<(), String> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;
        }
    }
    let json = serde_json::to_string_pretty(cases)
        .map_err(|e| format!("Failed to serialize regression cases: {}", e))?;
    fs::write(path, json.as_bytes())
        .map_err(|e| format!("Failed to write regression cases to {:?}: {}", path, e))
}

/// Sets up global service configuration for testing.
pub fn setup_test_service() {
    SERVICE_INIT.call_once(|| {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root_dir = if manifest_dir.ends_with("test") {
            manifest_dir.parent().unwrap().to_path_buf()
        } else {
            manifest_dir
        };

        let data_dir = root_dir.join("data");
        let model_dir = data_dir.join("models");
        let log_dir = data_dir.join("logs");

        let cfg = Config {
            log_level: "error".into(),
            config_dir: data_dir.clone(),
            model_dir,
            host: "127.0.0.1".into(),
            port: "8080".into(),
            enable_web_ui: false,
            enable_offline_mode: true,
            worker_idle_timeout: 3600,
            api_token: "".into(),
            log_dir,
            log_to_file: false,
            log_requests: false,
            max_sentence_length: 1024,
            max_request_body_size: 10 * 1024 * 1024,
            max_batch_size: 100,
            check_model_update: false,
            model_update_interval_secs: 0,
            cache_size: 2000,
            fullwidth_zh_punctuation: true,
            keep_models: vec![],
            keep_top_n: 0,
            max_loaded_models: 0,
            max_threads: 0,
            max_concurrency: 0,
            max_concurrency_per_ip: 0,
            trust_proxy: false,
            log_retention_days: 7,
            preprocess_symbols: false,
            parse_formats: ParseFormats::ALL,
            profile: false,
            bench_qps_input_length: 128,
        };

        config::set_for_testing(cfg);
        service::init();
        utils::cache::init();
        engine::rules::init_rules_from_config();
    });
}

/// Initializes models and service environment.
pub async fn init_test_environment() -> Result<(), String> {
    setup_test_service();
    models::init_records().await.map_err(|e| e.to_string())
}

/// Translates text using the real Marian translation service with automatic language detection.
pub async fn translate_case(target: &str, input: &str) -> Result<String, String> {
    service::translate_with_pivot("auto", target, input)
        .await
        .map_err(|e| format!("Translation failed: {:?}", e))
}

/// Deterministic mock translation for lightweight CI and unit testing without local models.
pub fn mock_translate_case(target: &str, input: &str) -> String {
    format!("[{}:{}]", target, input.trim())
}

/// Re-translates all cases in `path` and updates the `output` field in place.
pub async fn update_cases_file(path: impl AsRef<Path>, use_mock: bool) -> Result<usize, String> {
    let path = path.as_ref();
    let mut cases = load_cases(path)?;

    for case in &mut cases {
        let actual = if use_mock {
            mock_translate_case(&case.target, &case.input)
        } else {
            translate_case(&case.target, &case.input).await?
        };
        case.output = actual;
    }

    save_cases(path, &cases)?;
    Ok(cases.len())
}
