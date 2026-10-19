use std::path::PathBuf;
use std::sync::atomic::{AtomicPtr, Ordering};

use clap::builder::BoolishValueParser;
use clap::Parser;


/// Parsed command-line interface (also the library entry for tests/benches).
#[derive(Parser, Debug)]
#[command(name = "mtranserver", version)]
#[command(about = "MTranServer - Translation Server")]
pub struct Cli {
    /// Server host address
    #[arg(long, env = "MT_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Server port (1-65535)
    #[arg(long, env = "MT_PORT", default_value_t = 8989, value_parser = clap::value_parser!(u16).range(1..))]
    pub port: u16,

    /// Log level (debug, info, warn, error)
    #[arg(long, env = "MT_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Data directory (stores models, config, logs, and records)
    #[arg(long, env = "MT_DATA_DIR", default_value = "data")]
    pub data_dir: String,

    /// Enable offline mode. Bare `--offline` or `--offline=false` both work.
    #[arg(long, env = "MT_OFFLINE", default_value_t = false, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub offline: bool,

    /// Worker idle timeout in seconds
    #[arg(long, env = "MT_WORKER_IDLE_TIMEOUT", default_value_t = 60)]
    pub worker_idle_timeout: u64,

    /// API access token
    #[arg(long, env = "MT_API_TOKEN", default_value = "")]
    pub api_token: String,

    /// Cache size
    #[arg(long, env = "MT_CACHE_SIZE", default_value_t = 2000)]
    pub cache_size: usize,

    /// Fullwidth Chinese punctuation. Bare `--fullwidth-zh-punctuation` or
    /// `--fullwidth-zh-punctuation=false` both work.
    #[arg(long, env = "MT_FULLWIDTH_ZH_PUNCTUATION", default_value_t = true, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub fullwidth_zh_punctuation: bool,

    /// Preprocess source symbols before translation: rewrite in-word
    /// hyphens, camelCase, and snake_case splits into plain words so the
    /// model produces clean output for out-of-vocabulary identifiers.
    /// Applies only to tokens that look like identifiers; regular prose is
    /// left untouched.
    /// Disabled by default to align with official Firefox Translations policy.
    /// Bare `--preprocess-symbols` or `--preprocess-symbols=true` enables it;
    /// `--preprocess-symbols=false` disables it.
    #[arg(long, env = "MT_PREPROCESS_SYMBOLS", default_value_t = false, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub preprocess_symbols: bool,


    /// Markup formats to parse and preserve using word-alignment projection:
    /// 'html', 'md', 'html,md' (default), or 'false'/'0' to pass raw text to the model.
    #[arg(long, visible_alias = "formats", env = "MT_PARSE_FORMATS", default_value = "html,md", value_parser = parse_formats)]
    pub parse_formats: ParseFormats,
    /// Log HTTP requests (method, status, path, duration, bytes, client IP).
    /// Bare `--log-requests` or `--log-requests=false` both work.
    #[arg(long, env = "MT_LOG_REQUESTS", default_value_t = true, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub log_requests: bool,

    /// Write log files to the data directory. Bare `--log-to-file` or
    /// `--log-to-file=false` both work.
    #[arg(long, env = "MT_LOG_TO_FILE", default_value_t = false, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub log_to_file: bool,

    /// Maximum batch size (array length) for batch translation requests. 0 = unlimited.
    #[arg(long, env = "MT_MAX_BATCH_SIZE", default_value_t = 512)]
    pub max_batch_size: usize,

    /// Maximum sentence length; overlong text is auto-split into segments
    #[arg(long, env = "MT_MAX_SENTENCE_LENGTH", default_value_t = 512)]
    pub max_sentence_length: usize,

    /// Maximum request body size in bytes, or human-readable size (e.g. "512kb", "1mb").
    #[arg(long, env = "MT_MAX_REQUEST_BODY_SIZE", default_value = "512kb", value_parser = parse_size)]
    pub max_request_body_size: usize,

    /// Check for and auto-update model records and installed models.
    ///
    /// Options:
    /// - `false` / `0`: never check for updates (including startup records.json pull).
    /// - `true` / `1` or bare `--check-model-update`: check once asynchronously at startup.
    /// - duration such as `24h`, `1d`, `12` (hours): check asynchronously at startup
    ///   and repeat every that interval (units: `h` and `d`; default is `h`).
    ///
    /// Enabled by default (checks once asynchronously at startup; uses local records.json
    /// immediately to avoid blocking startup). The `MT_CHECK_MODEL_UPDATE` env var is also honored.
    #[arg(long, env = "MT_CHECK_MODEL_UPDATE", num_args = 0..=1, default_missing_value = "")]
    pub check_model_update: Option<Option<String>>,

    /// Keep specified language pairs resident in memory (never unloaded).
    /// Accepts multiple pairs, e.g. `--keep-models en_zh-Hans zh-Hans_en`,
    /// or a comma/space-separated `MT_KEEP_MODELS`. The special value
    /// `all` keeps every available model resident.
    #[arg(long, env = "MT_KEEP_MODELS", num_args = 1.., value_delimiter = ',')]
    pub keep_models: Vec<String>,

    /// Keep the N most-used models resident (in addition to --keep-models).
    /// Models fall off the kept set when their usage drops out of the top N.
    #[arg(long, env = "MT_KEEP_TOP_N", default_value_t = 0)]
    pub keep_top_n: usize,

    /// Maximum number of models held in memory at once. When a translation
    /// needs a model beyond this limit, the least-recently-used evictable
    /// model is unloaded immediately to make room. 0 = unlimited.
    #[arg(long, env = "MT_MAX_LOADED_MODELS", default_value_t = 0)]
    pub max_loaded_models: usize,

    /// Download models for specified pairs (space or comma separated).
    /// The special value `all` downloads every available language pair.
    #[arg(long, num_args = 1.., value_delimiter = ',')]
    pub download: Vec<String>,

    /// Maximum threads/cores for parallel sentence translation. 0 = all cores.
    #[arg(long, env = "MT_MAX_THREADS", default_value_t = 0)]
    pub max_threads: usize,

    /// Maximum concurrent requests allowed. 0 = unlimited.
    #[arg(long, env = "MT_MAX_CONCURRENCY", default_value_t = 0)]
    pub max_concurrency: usize,

    /// Maximum concurrent requests allowed per client IP. 0 = unlimited.
    #[arg(long, env = "MT_MAX_CONCURRENCY_PER_IP", default_value_t = 0)]
    pub max_concurrency_per_ip: usize,

    /// Trust client IP from X-Forwarded-For and X-Real-IP reverse proxy headers.
    #[arg(long, env = "MT_TRUST_PROXY", default_value_t = false, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub trust_proxy: bool,
    /// Log retention in days. 0 = do not automatically delete old logs.
    #[arg(long, env = "MT_LOG_RETENTION_DAYS", default_value_t = 7)]
    pub log_retention_days: usize,
    /// Enable web UI (embedded; visit `/`). Bare `--ui` or `--ui=false` both work.
    #[arg(long, env = "MT_ENABLE_UI", default_value_t = true, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub ui: bool,

    /// Print per-region engine timing (profiling) after translation activity.
    #[arg(long, env = "MT_PROFILE", default_value_t = false, num_args = 0..=1, default_missing_value = "true", value_parser = boolish())]
    pub profile: bool,

    /// List available language pairs
    #[arg(long)]
    pub languages: bool,

    /// Run a built-in QPS benchmark against a language pair. Defaults to `en_zh-Hans`.
    #[arg(
        long,
        value_name = "PAIR",
        num_args = 0..=1,
        default_missing_value = "en_zh-Hans"
    )]
    pub bench_qps: Option<String>,

    /// Input text length in characters for `--bench-qps`. Defaults to 128.
    #[arg(long, default_value_t = 128)]
    pub bench_qps_input_length: usize,
}

/// Bool parser accepting `true`/`false` (case-insensitive) and `1`/`0`, for
/// both CLI values and `MT_*` environment variables.
fn boolish() -> BoolishValueParser {
    BoolishValueParser::new()
}

/// Global server configuration, mirroring the original project's Config.
#[derive(Debug, Clone)]
pub struct Config {
    pub log_level: String,
    pub config_dir: PathBuf,
    pub model_dir: PathBuf,
    pub host: String,
    pub port: String,
    pub enable_web_ui: bool,
    pub enable_offline_mode: bool,
    pub worker_idle_timeout: u64,
    pub api_token: String,
    pub log_dir: PathBuf,
    pub log_to_file: bool,
    pub log_requests: bool,
    pub max_sentence_length: usize,
    /// Maximum request body size in bytes.
    pub max_request_body_size: usize,
    pub max_batch_size: usize,
    pub check_model_update: bool,
    /// Interval between update checks in seconds; `0` = check once at startup.
    pub model_update_interval_secs: u64,
    pub cache_size: usize,
    pub fullwidth_zh_punctuation: bool,
    /// Language pairs ("from-to") that stay resident forever.
    /// The special value "all" keeps every available model resident.
    pub keep_models: Vec<String>,
    /// Extra resident count by usage frequency (0 = disabled).
    pub keep_top_n: usize,
    /// Maximum simultaneously loaded models; 0 = unlimited. When exceeded,
    /// the least-recently-used evictable model is unloaded on demand.
    pub max_loaded_models: usize,
    /// Maximum threads/cores for parallel sentence translation. 0 = all cores.
    pub max_threads: usize,
    /// Maximum concurrent requests allowed; 0 = unlimited.
    pub max_concurrency: usize,
    /// Maximum concurrent requests allowed per client IP; 0 = unlimited.
    pub max_concurrency_per_ip: usize,
    pub trust_proxy: bool,
    pub log_retention_days: usize,
    /// Preprocess source symbols (hyphens, camelCase, snake_case) before
    /// translation. Disabled by default. See the `preprocess_symbols` CLI flag.
    pub preprocess_symbols: bool,
    /// Markup formats to parse. See the `parse_formats` CLI flag.
    pub parse_formats: ParseFormats,
    pub profile: bool,
    /// Input text length in characters for `--bench-qps`.
    pub bench_qps_input_length: usize,
}

/// Markup formats to parse and project using word alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ParseFormats {
    pub html: bool,
    pub md: bool,
}

impl Default for ParseFormats {
    fn default() -> Self {
        Self {
            html: true,
            md: true,
        }
    }
}

impl ParseFormats {
    pub const NONE: Self = Self {
        html: false,
        md: false,
    };

    pub const ALL: Self = Self {
        html: true,
        md: true,
    };

    pub fn has_html(&self) -> bool {
        self.html
    }

    pub fn has_md(&self) -> bool {
        self.md
    }

    pub fn is_empty(&self) -> bool {
        !self.html && !self.md
    }

    #[inline]
    pub const fn as_static_str(&self) -> &'static str {
        match (self.html, self.md) {
            (true, true) => "html,md",
            (true, false) => "html",
            (false, true) => "md",
            (false, false) => "",
        }
    }

    pub fn parse_from_json_value(val: &serde_json::Value) -> Result<Self, String> {
        match val {
            serde_json::Value::Bool(b) => Ok(if *b { Self::ALL } else { Self::NONE }),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    match i {
                        0 => Ok(Self::NONE),
                        1 => Ok(Self::ALL),
                        _ => Err(format!("invalid numeric format '{}': expected 0 or 1", i)),
                    }
                } else {
                    Err("invalid numeric format: expected integer 0 or 1".to_string())
                }
            }
            serde_json::Value::String(s) => s.parse::<Self>(),
            serde_json::Value::Null => Ok(Self::default()),
            other => Err(format!(
                "invalid parse format type: expected string, boolean, or integer, got {}",
                match other {
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => "object",
                    _ => "unknown",
                }
            )),
        }
    }
}

impl std::fmt::Display for ParseFormats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.html, self.md) {
            (true, true) => write!(f, "html,md"),
            (true, false) => write!(f, "html"),
            (false, true) => write!(f, "md"),
            (false, false) => write!(f, "false"),
        }
    }
}

impl std::str::FromStr for ParseFormats {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const MAX_PARSE_FORMAT_LEN: usize = 64;

        let trimmed = s.trim();
        if trimmed.len() > MAX_PARSE_FORMAT_LEN {
            return Err(format!(
                "parse format string exceeds maximum allowed length of {} bytes",
                MAX_PARSE_FORMAT_LEN
            ));
        }
        if trimmed.is_empty() {
            return Ok(Self::NONE);
        }

        // Stack buffer for in-place lowercase conversion without heap allocation
        let mut buf = [0u8; MAX_PARSE_FORMAT_LEN];
        let bytes = trimmed.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            buf[i] = b.to_ascii_lowercase();
        }
        let lower = std::str::from_utf8(&buf[..bytes.len()])
            .map_err(|_| "invalid utf-8 in parse format string".to_string())?;

        if matches!(lower, "false" | "0" | "none" | "off" | "no" | "disable") {
            return Ok(Self::NONE);
        }
        if matches!(lower, "true" | "1") {
            return Ok(Self::ALL);
        }

        let mut html = false;
        let mut md = false;

        for part in lower.split(|c| c == ',' || c == '+') {
            match part.trim() {
                "html" => html = true,
                "md" => md = true,
                other => {
                    return Err(format!(
                        "invalid format '{}': expected 'html', 'md', 'html,md', or 'false'/'0'",
                        other
                    ));
                }
            }
        }

        if !html && !md {
            return Err("invalid parse format: at least one format ('html' or 'md') must be specified".to_string());
        }

        Ok(Self { html, md })
    }
}

fn parse_formats(value: &str) -> Result<ParseFormats, String> {
    value.parse::<ParseFormats>()
}
/// Parse a human-readable size such as "512kb", "1mb", "1gb", or a raw byte count.
pub fn parse_size(value: &str) -> Result<usize, String> {
    let raw = value.trim();
    if raw.is_empty() {
        return Err("empty size string".to_string());
    }
    let lower = raw.to_ascii_lowercase();
    let (num_str, mult) =
        if let Some(s) = lower.strip_suffix("kb").or_else(|| lower.strip_suffix('k')) {
            (s, 1024usize)
        } else if let Some(s) = lower.strip_suffix("mb").or_else(|| lower.strip_suffix('m')) {
            (s, 1024 * 1024)
        } else if let Some(s) = lower.strip_suffix("gb").or_else(|| lower.strip_suffix('g')) {
            (s, 1024 * 1024 * 1024)
        } else if let Some(s) = lower.strip_suffix('b') {
            (s, 1)
        } else {
            (lower.as_str(), 1)
        };
    num_str
        .trim()
        .parse::<usize>()
        .map_err(|e| format!("invalid size '{}': {}", value, e))?
        .checked_mul(mult)
        .ok_or_else(|| format!("size overflow for '{}'", value))
}

/// Parse a boolean value accepted in environment variables: `true`/`false`
/// (case-insensitive) or `1`/`0`. Returns `None` on anything else.
pub fn parse_boolish(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

/// Clamp a numeric configuration value into `[min, max]`, warning on change.
fn clamp_range<T: PartialOrd + Copy + std::fmt::Display>(key: &str, value: T, min: T, max: T) -> T {
    if value < min {
        eprintln!(
            "[WARN] config: {} = {} is below minimum {}, clamping",
            key, value, min
        );
        min
    } else if value > max {
        eprintln!(
            "[WARN] config: {} = {} is above maximum {}, clamping",
            key, value, max
        );
        max
    } else {
        value
    }
}

/// Parse keep-model pairs ("from-to" or "from_to", space/comma separated)
/// into normalized "from-to" engine-pool keys. The special token `all`
/// (case-insensitive) is preserved verbatim: it marks every available model
/// as resident.
fn resolve_keep_models_pairs(raw: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for pair in raw {
        for token in pair.split([',', ' ']).filter(|s| !s.is_empty()) {
            if token.eq_ignore_ascii_case("all") {
                out.push("all".to_string());
                continue;
            }
            let (from, to) = match crate::utils::lang::split_language_pair(token) {
                Some((f, t)) => (f, t),
                None => continue,
            };
            let from = crate::utils::lang::normalize_language_code(from);
            let to = crate::utils::lang::normalize_language_code(to);
            out.push(format!("{}-{}", from, to));
        }
    }
    out
}

/// Resolve the model-update behavior.
///
/// Returns `(enabled, interval_secs)`. An interval of `0` means "check once
/// at startup" (the bare `--check-model-update` or `true` form); a positive value means
/// repeat every that many seconds. `false` disables all update checks.
/// Enabled by default (`(true, 0)`).
///
/// `MT_CHECK_MODEL_UPDATE` is parsed by clap into `cli_arg`, so all sources
/// arrive here uniformly: a bool-ish value (`true`/`1`, `false`/`0`),
/// a duration (`24h`), or the bare-flag sentinel `Some(Some(""))`.
fn resolve_model_update(cli_arg: Option<&Option<String>>) -> (bool, u64) {
    let Some(val) = cli_arg else {
        return (true, 0);
    };
    let raw = val.as_deref().unwrap_or("").trim();
    if let Some(b) = parse_boolish(raw) {
        return (b, 0);
    }
    (true, parse_interval_secs(Some(raw)).unwrap_or(0))
}

/// Parse a human-readable duration such as `24h`, `1d`, `12` into seconds.
/// Supports `d` (days) and `h` (hours). Without a unit suffix, it defaults to hours (`h`).
/// Returns `None` on garbage or unsupported units.
fn parse_interval_secs(value: Option<&str>) -> Option<u64> {
    let value = value?.trim();
    if value.is_empty() {
        return Some(0);
    }
    let (num, mult) = if let Some(s) = value.strip_suffix("d") {
        (s, 24 * 3600u64)
    } else if let Some(s) = value.strip_suffix("h") {
        (s, 3600u64)
    } else {
        (value, 3600u64) // bare number → hours
    };
    num.trim().parse::<u64>().ok().map(|n| n * mult)
}

/// Global server configuration pointer.
static CONFIG: AtomicPtr<Config> = AtomicPtr::new(std::ptr::null_mut());

/// Initialize the global config from CLI args and environment variables.
pub fn init(cli: Cli) {
    // `--data-dir` is the single root; model/config/log entries live under it.
    let data_dir = PathBuf::from(cli.data_dir);
    let config_dir = data_dir.clone();
    let model_dir = data_dir.join("models");
    let log_dir = data_dir.join("logs");

    // Range-check numeric settings with meaningful bounds; unsigned types
    // already reject negatives at the CLI/env parsing stage.
    let worker_idle_timeout = clamp_range(
        "MT_WORKER_IDLE_TIMEOUT",
        cli.worker_idle_timeout,
        1,
        u64::MAX,
    );
    let cache_size = clamp_range("MT_CACHE_SIZE", cli.cache_size, 0, usize::MAX);
    let keep_top_n = clamp_range("MT_KEEP_TOP_N", cli.keep_top_n, 0, usize::MAX);
    let max_loaded_models = clamp_range("MT_MAX_LOADED_MODELS", cli.max_loaded_models, 0, 4096);
    let max_threads = clamp_range("MT_MAX_THREADS", cli.max_threads, 0, 1024);
    let max_concurrency = clamp_range("MT_MAX_CONCURRENCY", cli.max_concurrency, 0, usize::MAX);
    let max_concurrency_per_ip = clamp_range(
        "MT_MAX_CONCURRENCY_PER_IP",
        cli.max_concurrency_per_ip,
        0,
        usize::MAX,
    );
    let log_retention_days = clamp_range("MT_LOG_RETENTION_DAYS", cli.log_retention_days, 0, 36500);
    let max_sentence_length =
        clamp_range("MT_MAX_SENTENCE_LENGTH", cli.max_sentence_length, 16, 8192);
    let max_request_body_size = clamp_range(
        "MT_MAX_REQUEST_BODY_SIZE",
        cli.max_request_body_size,
        1024,
        usize::MAX,
    );
    let max_batch_size = clamp_range("MT_MAX_BATCH_SIZE", cli.max_batch_size, 0, usize::MAX);
    // Resolve the model-update behavior once; both fields come from the same
    // (env-var-reading) call.
    let (check_model_update, model_update_interval_secs) =
        resolve_model_update(cli.check_model_update.as_ref());

    let cfg = Config {
        log_level: cli.log_level,
        config_dir,
        model_dir,
        host: cli.host,
        port: cli.port.to_string(),
        enable_web_ui: cli.ui,
        enable_offline_mode: cli.offline,
        worker_idle_timeout,
        api_token: cli.api_token,
        log_dir,
        log_to_file: cli.log_to_file,
        log_retention_days,
        log_requests: cli.log_requests,
        max_sentence_length,
        max_request_body_size,
        max_batch_size,
        check_model_update,
        model_update_interval_secs,
        cache_size,
        fullwidth_zh_punctuation: cli.fullwidth_zh_punctuation,
        keep_models: resolve_keep_models_pairs(&cli.keep_models),
        keep_top_n,
        max_loaded_models,
        max_threads,
        max_concurrency,
        max_concurrency_per_ip,
        trust_proxy: cli.trust_proxy,
        preprocess_symbols: cli.preprocess_symbols,
        parse_formats: cli.parse_formats,
        profile: cli.profile,
        bench_qps_input_length: cli.bench_qps_input_length,
    };

    let ptr = Box::into_raw(Box::new(cfg));
    #[cfg(not(test))]
    {
        if CONFIG
            .compare_exchange(
                std::ptr::null_mut(),
                ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // SAFETY: `ptr` was created by `Box::into_raw` above and never exposed.
            let _ = unsafe { Box::from_raw(ptr) };
            panic!("config::init called twice in the same process");
        }
    }
    #[cfg(test)]
    {
        CONFIG.swap(ptr, Ordering::AcqRel);
    }
}

/// Check whether the global config has been initialized.
pub fn is_initialized() -> bool {
    !CONFIG.load(Ordering::Acquire).is_null()
}

/// Get the global config. Panics if not initialized.
pub fn get() -> &'static Config {
    let ptr = CONFIG.load(Ordering::Acquire);
    if ptr.is_null() {
        panic!("Config not initialized. Call config::init() first.");
    }
    // SAFETY: The pointer is initialized from a leaked Box and remains valid for the static lifetime.
    unsafe { &*ptr }
}

#[doc(hidden)]
pub fn set_for_testing(cfg: Config) {
    let ptr = Box::into_raw(Box::new(cfg));
    CONFIG.swap(ptr, Ordering::AcqRel);
}

#[cfg(test)]
pub(crate) static TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_valid() {
        assert_eq!(parse_size("0").unwrap(), 0);
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1024b").unwrap(), 1024);
        assert_eq!(parse_size("1024B").unwrap(), 1024);
        assert_eq!(parse_size("512k").unwrap(), 512 * 1024);
        assert_eq!(parse_size("512kb").unwrap(), 512 * 1024);
        assert_eq!(parse_size("512KB").unwrap(), 512 * 1024);
        assert_eq!(parse_size("512Kb").unwrap(), 512 * 1024);
        assert_eq!(parse_size("  512 kb  ").unwrap(), 512 * 1024);
        assert_eq!(parse_size("1m").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("1mb").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("1MB").unwrap(), 1024 * 1024);
        assert_eq!(parse_size("2g").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("2gb").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("2GB").unwrap(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_size_invalid() {
        assert!(parse_size("").is_err());
        assert!(parse_size("   ").is_err());
        assert!(parse_size("abc").is_err());
        assert!(parse_size("12abc").is_err());
        assert!(parse_size("-10").is_err());
        assert!(parse_size("1000000000000000000000000000gb").is_err());
    }

    #[test]
    fn test_bench_qps_cli_parsing() {
        let cli_none = Cli::try_parse_from(["mtranserver"]).unwrap();
        assert_eq!(cli_none.bench_qps, None);

        let cli_default = Cli::try_parse_from(["mtranserver", "--bench-qps"]).unwrap();
        assert_eq!(cli_default.bench_qps, Some("en_zh-Hans".to_string()));

        let cli_custom = Cli::try_parse_from(["mtranserver", "--bench-qps", "fr_en"]).unwrap();
        assert_eq!(cli_custom.bench_qps, Some("fr_en".to_string()));
    }
    #[test]
    fn test_bench_qps_input_length_cli_parsing() {
        let cli_default = Cli::try_parse_from(["mtranserver"]).unwrap();
        assert_eq!(cli_default.bench_qps_input_length, 128);

        let cli_custom =
            Cli::try_parse_from(["mtranserver", "--bench-qps-input-length", "256"]).unwrap();
        assert_eq!(cli_custom.bench_qps_input_length, 256);
    }

    #[test]
    fn test_check_model_update_cli_parsing() {
        let cli_none = Cli::try_parse_from(["mtranserver"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_none.check_model_update.as_ref()),
            (true, 0)
        );

        let cli_bare = Cli::try_parse_from(["mtranserver", "--check-model-update"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_bare.check_model_update.as_ref()),
            (true, 0)
        );

        let cli_true =
            Cli::try_parse_from(["mtranserver", "--check-model-update", "true"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_true.check_model_update.as_ref()),
            (true, 0)
        );

        let cli_eq_true =
            Cli::try_parse_from(["mtranserver", "--check-model-update=true"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_eq_true.check_model_update.as_ref()),
            (true, 0)
        );

        let cli_false =
            Cli::try_parse_from(["mtranserver", "--check-model-update", "false"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_false.check_model_update.as_ref()),
            (false, 0)
        );

        let cli_eq_false =
            Cli::try_parse_from(["mtranserver", "--check-model-update=false"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_eq_false.check_model_update.as_ref()),
            (false, 0)
        );

        let cli_24h = Cli::try_parse_from(["mtranserver", "--check-model-update", "24h"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_24h.check_model_update.as_ref()),
            (true, 86400)
        );

        let cli_1d = Cli::try_parse_from(["mtranserver", "--check-model-update", "1d"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_1d.check_model_update.as_ref()),
            (true, 86400)
        );

        let cli_12 = Cli::try_parse_from(["mtranserver", "--check-model-update", "12"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_12.check_model_update.as_ref()),
            (true, 43200)
        );

        // 'm' unit is not supported, treated as invalid duration -> fallback interval 0 (startup only)
        let cli_eq_30m = Cli::try_parse_from(["mtranserver", "--check-model-update=30m"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_eq_30m.check_model_update.as_ref()),
            (true, 0)
        );

        let cli_0 = Cli::try_parse_from(["mtranserver", "--check-model-update", "0"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_0.check_model_update.as_ref()),
            (false, 0)
        );

        let cli_1 = Cli::try_parse_from(["mtranserver", "--check-model-update", "1"]).unwrap();
        assert_eq!(
            resolve_model_update(cli_1.check_model_update.as_ref()),
            (true, 0)
        );
    }

    #[test]
    fn test_max_batch_size_cli_parsing() {
        let cli_default = Cli::try_parse_from(["mtranserver"]).unwrap();
        assert_eq!(cli_default.max_batch_size, 512);

        let cli_custom = Cli::try_parse_from(["mtranserver", "--max-batch-size", "10"]).unwrap();
        assert_eq!(cli_custom.max_batch_size, 10);

        let cli_zero = Cli::try_parse_from(["mtranserver", "--max-batch-size", "0"]).unwrap();
        assert_eq!(cli_zero.max_batch_size, 0);
    }

    #[test]
    fn test_global_config_lifecycle() {
        let _guard = TEST_LOCK.lock();
        let cli = Cli::try_parse_from(["mtranserver"]).unwrap();
        init(cli);
        assert!(is_initialized());
        let cfg = get();
        assert_eq!(cfg.port, "8989");

        let mut test_cfg = cfg.clone();
        test_cfg.port = "9999".to_string();
        set_for_testing(test_cfg);
        assert!(is_initialized());
        assert_eq!(get().port, "9999");
    }
    #[test]
    fn test_parse_formats_parsing() {
        assert_eq!("html".parse::<ParseFormats>().unwrap(), ParseFormats { html: true, md: false });
        assert_eq!("HTML".parse::<ParseFormats>().unwrap(), ParseFormats { html: true, md: false });
        assert_eq!("md".parse::<ParseFormats>().unwrap(), ParseFormats { html: false, md: true });
        assert_eq!("MD".parse::<ParseFormats>().unwrap(), ParseFormats { html: false, md: true });
        assert_eq!("html,md".parse::<ParseFormats>().unwrap(), ParseFormats { html: true, md: true });
        assert_eq!("HTML,MD".parse::<ParseFormats>().unwrap(), ParseFormats { html: true, md: true });
        assert_eq!("Md,Html".parse::<ParseFormats>().unwrap(), ParseFormats { html: true, md: true });
        assert!("all".parse::<ParseFormats>().is_err());
        assert_eq!("0".parse::<ParseFormats>().unwrap(), ParseFormats::NONE);
        assert_eq!("none".parse::<ParseFormats>().unwrap(), ParseFormats::NONE);
        assert!("mod".parse::<ParseFormats>().is_err());
        assert!("markdown".parse::<ParseFormats>().is_err());
        assert!("invalid".parse::<ParseFormats>().is_err());
        assert!("html,".parse::<ParseFormats>().is_err());
        assert!(",html".parse::<ParseFormats>().is_err());
        assert!("html,invalid".parse::<ParseFormats>().is_err());

        // Injection attack / oversized string protection
        let oversized = "a".repeat(65);
        assert!(oversized.parse::<ParseFormats>().is_err());
        let oversized_huge = "html,".repeat(1000);
        assert!(oversized_huge.parse::<ParseFormats>().is_err());

        // Test parse_from_json_value
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::json!("html")).unwrap(), ParseFormats { html: true, md: false });
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::json!("MD")).unwrap(), ParseFormats { html: false, md: true });
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::json!(false)).unwrap(), ParseFormats::NONE);
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::json!(0)).unwrap(), ParseFormats::NONE);
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::json!("false")).unwrap(), ParseFormats::NONE);
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::json!("0")).unwrap(), ParseFormats::NONE);
        assert_eq!(ParseFormats::parse_from_json_value(&serde_json::Value::Null).unwrap(), ParseFormats::default());
        assert!(ParseFormats::parse_from_json_value(&serde_json::json!("mod")).is_err());
        assert!(ParseFormats::parse_from_json_value(&serde_json::json!(2)).is_err());
        assert!(ParseFormats::parse_from_json_value(&serde_json::json!(0.5)).is_err());
        assert!(ParseFormats::parse_from_json_value(&serde_json::json!(["html"])).is_err());
        assert!(ParseFormats::parse_from_json_value(&serde_json::json!({"html": true})).is_err());
        assert!(ParseFormats::parse_from_json_value(&serde_json::json!("a".repeat(100))).is_err());
        let cli_default = Cli::try_parse_from(["mtranserver"]).unwrap();
        assert_eq!(cli_default.parse_formats, ParseFormats::ALL);
        let cli_html = Cli::try_parse_from(["mtranserver", "--parse-formats", "html"]).unwrap();
        assert_eq!(cli_html.parse_formats, ParseFormats { html: true, md: false });
        let cli_false = Cli::try_parse_from(["mtranserver", "--parse-formats", "false"]).unwrap();
        assert_eq!(cli_false.parse_formats, ParseFormats::NONE);
        let cli_alias = Cli::try_parse_from(["mtranserver", "--formats=md"]).unwrap();
        assert_eq!(cli_alias.parse_formats, ParseFormats { html: false, md: true });
    }

    #[test]
    fn test_preprocess_symbols_cli_parsing() {
        let cli_default = Cli::try_parse_from(["mtranserver"]).unwrap();
        assert!(!cli_default.preprocess_symbols);

        let cli_bare = Cli::try_parse_from(["mtranserver", "--preprocess-symbols"]).unwrap();
        assert!(cli_bare.preprocess_symbols);

        let cli_true = Cli::try_parse_from(["mtranserver", "--preprocess-symbols=true"]).unwrap();
        assert!(cli_true.preprocess_symbols);

        let cli_false = Cli::try_parse_from(["mtranserver", "--preprocess-symbols=false"]).unwrap();
        assert!(!cli_false.preprocess_symbols);
    }

}
