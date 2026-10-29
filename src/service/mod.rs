//! Translation service layer.
//!
//! Manages the engine pool (one engine per language pair), translation with
//! pivot routing, caching, and Chinese punctuation formatting. Mirrors
//! `src/services/engine.ts` in the original project.

pub mod detector;
pub mod error;

pub use error::ServiceError;

/// Structured result of a single-text translation operation.
#[derive(Debug, Clone)]
pub struct TranslationOutput {
    /// Translated result, preserving zero-copy reference when obtained from cache.
    pub text: TranslatedText,
    /// Detected source language if automatic detection was requested, or the input language.
    pub detected_language: String,
    /// Duration consumed by the translation step itself.
    pub translation_duration: std::time::Duration,
    /// Duration consumed by loading missing model weights, if any.
    pub model_load_duration: std::time::Duration,
}

impl TranslationOutput {
    /// Borrow the translated text as a string slice without heap allocations.
    #[inline]
    pub fn text_as_str(&self) -> &str {
        self.text.as_str()
    }

    /// Consume the output and return an owned `String`.
    #[inline]
    pub fn into_text_string(self) -> String {
        self.text.into_string()
    }
}

/// Represents the result of a translation step, either an owned `String`
/// or a zero-copy reference to a cached result `Arc<str>`.
///
/// Implements `Deref<Target = str>`, `AsRef<str>`, `Display`, `Serialize`,
/// and `PartialEq` with both `str` and `String`, allowing caller ergonomics
/// identical to string types while preserving zero-copy cached lookups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslatedText {
    /// Freshly translated owned string.
    String(String),
    /// Zero-copy shared reference to a cached translation result.
    Cached(Arc<str>),
}

impl Default for TranslatedText {
    #[inline]
    fn default() -> Self {
        TranslatedText::String(String::new())
    }
}

impl TranslatedText {
    /// Borrow the translated text as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        match self {
            TranslatedText::String(s) => s.as_str(),
            TranslatedText::Cached(s) => s.as_ref(),
        }
    }

    /// Convert into an owned `String`, cloning only if backed by a cached `Arc<str>`.
    #[inline]
    pub fn into_string(self) -> String {
        match self {
            TranslatedText::String(s) => s,
            TranslatedText::Cached(s) => s.to_string(),
        }
    }

    /// Apply typographic normalization and truecasing, reusing the existing backing storage when no modifications are made.
    #[inline]
    pub fn normalize_punctuation(self, to_lang: &str) -> TranslatedText {
        let text_cow = std::borrow::Cow::Borrowed(self.as_str());
        let processed = postprocess_typography(text_cow, to_lang);
        match processed {
            std::borrow::Cow::Borrowed(_) => self,
            std::borrow::Cow::Owned(s) => TranslatedText::String(s),
        }
    }
}

impl std::ops::Deref for TranslatedText {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for TranslatedText {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::borrow::Borrow<str> for TranslatedText {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for TranslatedText {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl serde::Serialize for TranslatedText {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for TranslatedText {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer).map(TranslatedText::String)
    }
}

impl From<String> for TranslatedText {
    #[inline]
    fn from(s: String) -> Self {
        TranslatedText::String(s)
    }
}

impl From<&str> for TranslatedText {
    #[inline]
    fn from(s: &str) -> Self {
        TranslatedText::String(s.to_string())
    }
}

impl From<Arc<str>> for TranslatedText {
    #[inline]
    fn from(s: Arc<str>) -> Self {
        TranslatedText::Cached(s)
    }
}

impl From<TranslatedText> for String {
    #[inline]
    fn from(t: TranslatedText) -> Self {
        t.into_string()
    }
}

impl PartialEq<str> for TranslatedText {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for TranslatedText {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<TranslatedText> for &str {
    #[inline]
    fn eq(&self, other: &TranslatedText) -> bool {
        *self == other.as_str()
    }
}

impl PartialEq<TranslatedText> for str {
    #[inline]
    fn eq(&self, other: &TranslatedText) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<String> for TranslatedText {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<TranslatedText> for String {
    #[inline]
    fn eq(&self, other: &TranslatedText) -> bool {
        self.as_str() == other.as_str()
    }
}

/// Apply typography normalization to translated text, preserving cached references when unchanged.
#[inline]
pub fn normalize_output_punctuation(
    text: impl Into<TranslatedText>,
    to_lang: &str,
) -> TranslatedText {
    text.into().normalize_punctuation(to_lang)
}
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use futures::StreamExt;
use tokio::time::{sleep, Duration};

use parking_lot::{Mutex, RwLock};

use crate::config;
use crate::engine::{EngineConfig, TranslationEngine};
use crate::models;

pub use crate::engine::EngineUsage;

struct EngineInfo {
    engine: Arc<TranslationEngine>,
    idle_task: Option<tokio::task::JoinHandle<()>>,
}

impl EngineInfo {
    #[inline]
    fn now_ms() -> u64 {
        crate::engine::now_ms()
    }

    #[inline]
    fn idle_for(&self, timeout: Duration) -> bool {
        self.engine.idle_for(timeout)
    }

    #[inline]
    fn last_used_ms(&self) -> u64 {
        self.engine.last_used_ms()
    }
}

/// Global engine pool keyed by "from-to".
static ENGINES: RwLock<Option<HashMap<String, EngineInfo>>> = RwLock::new(None);

type LoadingPromise = Arc<tokio::sync::Mutex<Option<Result<Arc<TranslationEngine>, ServiceError>>>>;

/// Pending engine load promises keyed by "from-to".
static LOADING: Mutex<Option<HashMap<String, LoadingPromise>>> = Mutex::new(None);

/// Global concurrency gate for CPU-bound engine calls.
///
/// Every blocking `engine.translate` call acquires one permit first, so N
/// concurrent requests can never run more than `max_threads` (0 = all
/// logical cores) GEMMs at once — the tokio blocking pool's own limit (512)
/// would otherwise oversubscribe the CPU and tank tail latency. Acquired on
/// the async side *before* entering `spawn_blocking`; the pivot path takes
/// the two hops sequentially, so permits never nest.
static ENGINE_GATE: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();

fn engine_gate() -> &'static tokio::sync::Semaphore {
    ENGINE_GATE.get_or_init(|| {
        let max_threads = config::get().max_threads;
        let permits = if max_threads > 0 {
            max_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .max(1)
        };
        tokio::sync::Semaphore::new(permits)
    })
}

pub fn init() {
    *ENGINES.write() = Some(HashMap::new());
    *LOADING.lock() = Some(HashMap::new());
}

fn key(from: &str, to: &str) -> String {
    format!("{}-{}", from, to)
}

#[inline]
fn with_key<R>(from: &str, to: &str, f: impl FnOnce(&str) -> R) -> R {
    let len = from.len() + 1 + to.len();
    if len <= 64 {
        let mut buf = [0u8; 64];
        buf[..from.len()].copy_from_slice(from.as_bytes());
        buf[from.len()] = b'-';
        buf[from.len() + 1..len].copy_from_slice(to.as_bytes());
        let s = unsafe { std::str::from_utf8_unchecked(&buf[..len]) };
        f(s)
    } else {
        f(&key(from, to))
    }
}

/// In-memory usage counters: engine key "from-to" -> engine usage stats.
static USAGE: std::sync::LazyLock<RwLock<HashMap<String, Arc<EngineUsage>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

#[allow(dead_code)]
pub(crate) fn get_or_create_usage(k: &str) -> Arc<EngineUsage> {
    {
        let guard = USAGE.read();
        if let Some(usage) = guard.get(k) {
            return usage.clone();
        }
    }
    let mut guard = USAGE.write();
    guard
        .entry(k.to_string())
        .or_insert_with(|| Arc::new(EngineUsage::new()))
        .clone()
}

/// Record one use of an engine key.
///
/// The counters only feed top-N residency scoring; when `keep_top_n == 0` no
/// score is ever consulted, so skip the lock and the key allocation entirely.
#[allow(dead_code)]
pub(crate) fn record_usage(k: &str) {
    if config::get().keep_top_n == 0 {
        return;
    }
    get_or_create_usage(k).record_call();
}

/// True if `--keep-models` was given (pinning everything when the special
/// value "all" is present, or the exact pair otherwise).
fn is_pinned(k: &str) -> bool {
    let cfg = config::get();
    cfg.keep_models.iter().any(|m| m == "all" || m == k)
}

/// True if the key currently ranks within the top N used models.
fn is_top_n(k: &str) -> bool {
    let cfg = config::get();
    if cfg.keep_top_n == 0 {
        return false;
    }
    let n = cfg.keep_top_n;
    let (mut counts, target_count) = {
        let guard = USAGE.read();
        // Count distinct scored keys; if fewer than N, everything resident is kept.
        let count = guard.len();
        if count == 0 {
            return false;
        }
        // If the map has <= N entries, any recorded key with activity qualifies.
        if count <= n {
            return guard.get(k).map(|c| c.call_count() > 0).unwrap_or(false);
        }
        let target = match guard.get(k) {
            Some(c) => c.call_count(),
            None => return false,
        };
        if target == 0 {
            return false;
        }
        let mut counts = Vec::with_capacity(count);
        for val in guard.values() {
            counts.push(val.call_count());
        }
        (counts, target)
    };
    // O(m log m) sort — this runs under the idle-task path on every cycle).
    let threshold_idx = n - 1;
    counts.select_nth_unstable_by(threshold_idx, |a, b| b.cmp(a));
    let threshold = counts[threshold_idx];
    target_count > 0 && target_count >= threshold
}
/// True if the engine must stay resident (pinned, or currently in top N).
fn should_keep_resident(k: &str) -> bool {
    is_pinned(k) || is_top_n(k)
}

/// Warm up every `--keep-models` pair at startup so the first request to a
/// pinned model never pays the cold start. Runs after the server binds its
/// listener, in the background; failures are logged and skipped (the normal
/// request path will surface them again with full context).
///
/// Keys arrive normalized as `from-to` by `resolve_keep_models_pairs`.
/// Language pairs are split using `split_language_pair` so hyphenated codes
/// (e.g. `zh-Hans-en`, `en-zh-Hans`) are correctly recognized.
/// The special value `all` expands to every language pair in the model
/// records before preloading.
pub fn preload_engines() {
    let mut pairs = config::get().keep_models.clone();
    if pairs.is_empty() {
        return;
    }
    if pairs.iter().any(|p| p == "all") {
        pairs = models::get_language_pairs();
    }
    tokio::spawn(async move {
        for pair in pairs {
            let Some((from, to)) = crate::utils::lang::split_language_pair(&pair) else {
                continue;
            };
            match get_or_create_engine(from, to).await {
                Ok(_) => tracing::info!("Preloaded model for {} -> {}", from, to),
                Err(e) => {
                    tracing::warn!("Failed to preload model for {} -> {}: {}", from, to, e)
                }
            }
        }
    });
}

/// Check whether a language pair needs pivot translation via English.
fn needs_pivot_translation(from_lang: &str, to_lang: &str) -> bool {
    if from_lang == "en" || to_lang == "en" {
        return false;
    }
    if models::has_language_pair(from_lang, to_lang) {
        return false;
    }
    true
}

/// Get or create the engine for a language pair (loads model if needed).
struct LoadingCleanupGuard<'a> {
    key: &'a str,
    promise: &'a LoadingPromise,
}

impl<'a> Drop for LoadingCleanupGuard<'a> {
    fn drop(&mut self) {
        let mut guard = LOADING.lock();
        if let Some(map) = guard.as_mut() {
            if map.get(self.key).is_some_and(|p| Arc::ptr_eq(p, self.promise)) {
                map.remove(self.key);
            }
        }
    }
}

async fn get_or_create_engine(
    from_lang: &str,
    to_lang: &str,
) -> Result<Arc<TranslationEngine>, ServiceError> {
    let k = key(from_lang, to_lang);

    loop {
        // Fast path: engine already loaded — re-arm its idle timer.
        if let Some(engine) = reuse_existing_engine(&k) {
            return Ok(engine);
        }

        enum LoadSlot {
            Loader(
                LoadingPromise,
                tokio::sync::OwnedMutexGuard<Option<Result<Arc<TranslationEngine>, ServiceError>>>,
            ),
            Waiter(LoadingPromise),
        }

        let slot = {
            let mut guard = LOADING.lock();
            let map = guard.get_or_insert_with(HashMap::new);
            match map.entry(k.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let promise = Arc::new(tokio::sync::Mutex::new(None));
                    let loader_guard = promise
                        .clone()
                        .try_lock_owned()
                        .expect("uncontended lock");
                    entry.insert(promise.clone());
                    LoadSlot::Loader(promise, loader_guard)
                }
                std::collections::hash_map::Entry::Occupied(entry) => {
                    LoadSlot::Waiter(entry.get().clone())
                }
            }
        };

        match slot {
            LoadSlot::Loader(promise, mut loader_guard) => {
                let cleanup_guard = LoadingCleanupGuard {
                    key: &k,
                    promise: &promise,
                };
                let result = load_and_register_engine(from_lang, to_lang, &k).await;
                *loader_guard = Some(result.clone());
                drop(cleanup_guard);
                drop(loader_guard);
                return result;
            }
            LoadSlot::Waiter(promise) => {
                let loaded = promise.lock().await;
                match loaded.as_ref() {
                    Some(Ok(engine)) => {
                        reuse_existing_engine(&k);
                        return Ok(engine.clone());
                    }
                    Some(Err(err)) => return Err(err.clone()),
                    None => {
                        // Previous loader was cancelled before completing; retry.
                        continue;
                    }
                }
            }
        }
    }
}

fn reuse_existing_engine(k: &str) -> Option<Arc<TranslationEngine>> {
    let guard = ENGINES.read();
    let map = guard.as_ref()?;
    let info = map.get(k)?;
    info.engine.touch();
    Some(info.engine.clone())
}

/// Engines prepared for a translation route.
#[derive(Clone)]
pub enum PreparedEngines {
    SameLanguage,
    Direct(Arc<TranslationEngine>),
    Pivot(Arc<TranslationEngine>, Arc<TranslationEngine>),
}

impl std::fmt::Debug for PreparedEngines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreparedEngines::SameLanguage => write!(f, "PreparedEngines::SameLanguage"),
            PreparedEngines::Direct(_) => write!(f, "PreparedEngines::Direct(...)"),
            PreparedEngines::Pivot(_, _) => write!(f, "PreparedEngines::Pivot(...)"),
        }
    }
}

impl PreparedEngines {
    #[inline]
    pub fn is_same_language(&self) -> bool {
        matches!(self, PreparedEngines::SameLanguage)
    }
}

/// Ensure the engine for a language pair is loaded into memory.
///
/// Returns the engine and the time taken to load it (or `Duration::ZERO` if already resident).
pub async fn ensure_engine_loaded(
    from_lang: &str,
    to_lang: &str,
) -> Result<(Arc<TranslationEngine>, Duration), ServiceError> {
    if let Some(engine) = with_key(from_lang, to_lang, reuse_existing_engine) {
        return Ok((engine, Duration::ZERO));
    }
    let start = std::time::Instant::now();
    let max_retries = 3usize;
    let mut last_error: Option<ServiceError> = None;

    for attempt in 0..max_retries {
        match get_or_create_engine(from_lang, to_lang).await {
            Ok(engine) => return Ok((engine, start.elapsed())),
            Err(e) => {
                if !e.is_retryable() {
                    return Err(e);
                }
                if attempt == max_retries - 1 {
                    return Err(ServiceError::Internal(format!(
                        "Failed to load engine after {} retries: {}",
                        max_retries, e
                    )));
                }
                last_error = Some(e);
                tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1))).await;
            }
        }
    }

    Err(ServiceError::Internal(format!(
        "Failed to load engine after {} retries: {}",
        max_retries,
        last_error.map_or_else(|| "unknown error".to_string(), |e| e.to_string())
    )))
}

/// Ensure all required engines (direct or pivot pair) are loaded into memory.
///
/// If pivot routing is needed, both engines are loaded concurrently using `tokio::join!`.
/// Returns the prepared engines and total model load duration (or `Duration::ZERO` if already loaded).
pub async fn ensure_required_engines(
    from_lang: &str,
    to_lang: &str,
) -> Result<(PreparedEngines, Duration), ServiceError> {
    if from_lang == to_lang {
        return Ok((PreparedEngines::SameLanguage, Duration::ZERO));
    }
    if !needs_pivot_translation(from_lang, to_lang) {
        let (engine, d) = ensure_engine_loaded(from_lang, to_lang).await?;
        Ok((PreparedEngines::Direct(engine), d))
    } else {
        let (res1, res2) = tokio::join!(
            ensure_engine_loaded(from_lang, "en"),
            ensure_engine_loaded("en", to_lang)
        );
        let (engine1, d1) = res1?;
        let (engine2, d2) = res2?;
        Ok((PreparedEngines::Pivot(engine1, engine2), d1.max(d2)))
    }
}
/// Spawn the persistent residency-check task for one engine key.
///
/// Spawned once per engine (never aborted/rearmed on use), it wakes every
/// `worker_idle_timeout` and evicts the engine only when it has been idle for
/// a full period and may not stay resident:
/// - explicit `--keep-models` pins: always survive the cycle (resident).
/// - top-N residents survive while they keep their ranking; the moment usage
///   drops them out of the top N they are evicted on the next cycle —
///   i.e. delayed unload follows `worker-idle-timeout`, as required.
/// - everything else is evicted one interval after its last use.
///
/// The task exits by itself once its key is no longer in the pool.
fn spawn_idle_task(k: String) -> tokio::task::JoinHandle<()> {
    let timeout_secs = config::get().worker_idle_timeout;
    tokio::spawn(async move {
        let timeout = Duration::from_secs(timeout_secs);
        loop {
            sleep(timeout).await;
            // Residency is decided from the atomic timestamp and the usage
            let (idle, resident) = {
                let guard = ENGINES.read();
                match guard.as_ref().and_then(|m| m.get(&k)) {
                    None => break, // engine already gone — stop watching
                    Some(info) => (info.idle_for(timeout), should_keep_resident(&k)),
                }
            };
            if idle && !resident {
                let mut guard = ENGINES.write();
                // Re-check under the lock: another request may have touched
                // or replaced the entry between the two critical sections.
                let still_evictable = match guard.as_mut().and_then(|m| m.get_mut(&k)) {
                    None => true,
                    Some(info) => info.idle_for(timeout) && !should_keep_resident(&k),
                };
                if still_evictable {
                    guard.as_mut().map(|m| m.remove(&k));
                    break;
                }
            }
        }
    })
}

/// Download (if needed), create, and register the engine for a language pair.
async fn load_and_register_engine(
    from_lang: &str,
    to_lang: &str,
    k: &str,
) -> Result<Arc<TranslationEngine>, ServiceError> {
    let cfg = config::get();

    // Mirror TS: in online mode check the model registry first. A pair missing
    // from the records fails fast with "No model found" (never a misleading
    // "Model file not found", which implies a download failure).
    if !cfg.enable_offline_mode && !models::has_language_pair(from_lang, to_lang) {
        return Err(ServiceError::ModelNotFound(format!(
            "No model found for {} -> {}",
            from_lang, to_lang
        )));
    }

    let model_files = match models::get_model_files(&cfg.model_dir, from_lang, to_lang).await {
        Ok(files) => files,
        Err(e) => {
            // Auto-download the model when not in offline mode, then retry.
            if !cfg.enable_offline_mode {
                tracing::info!(
                    "Model missing for {} -> {}, downloading...",
                    from_lang,
                    to_lang
                );
                models::download_model(from_lang, to_lang, None)
                    .await
                    .map_err(ServiceError::transient)?;
                models::get_model_files(&cfg.model_dir, from_lang, to_lang)
                    .await
                    .map_err(ServiceError::ModelNotFound)?
            } else {
                return Err(ServiceError::ModelNotFound(e));
            }
        }
    };

    let model = PathBuf::from(&model_files["model"]);
    let lex = PathBuf::from(&model_files["lex"]);
    let vocab_src = PathBuf::from(&model_files["vocab_src"]);
    let vocab_trg = PathBuf::from(&model_files["vocab_trg"]);
    let src = from_lang.to_string();
    let trg = to_lang.to_string();

    // Parsing model/spm/shortlist files is hundreds of milliseconds of
    // synchronous CPU+IO work — run it on the blocking pool so async workers
    // stay responsive during cold starts.
    let engine = match tokio::task::spawn_blocking(move || {
        TranslationEngine::new(
            &src,
            &trg,
            &model,
            &lex,
            &vocab_src,
            &vocab_trg,
            if config::is_initialized() {
                EngineConfig::from_config(&config::get())
            } else {
                EngineConfig::default()
            },
        )
        .map_err(ServiceError::Engine)
    })
    .await
    {
        Ok(Ok(engine)) => engine,
        Ok(Err(e)) => return Err(e),
        Err(e) => {
            return Err(ServiceError::Internal(format!(
                "Engine loader panicked: {}",
                e
            )))
        }
    };

    let usage = get_or_create_usage(k);
    let mut engine = engine;
    engine.set_usage(usage);
    let engine = Arc::new(engine);

    // Every engine gets the periodic residency-check task. Pinned and top-N
    // models survive each cycle (resident); a model that falls out of the top
    // N is unloaded on the next cycle — delayed by worker-idle-timeout.
    let idle_task = spawn_idle_task(k.to_string());
    {
        let mut guard = ENGINES.write();
        if let Some(map) = guard.as_mut() {
            // Capacity limit: make room for the new engine before inserting
            // by immediately unloading the least-recently-used evictable
            // model. `keep-models` pins and top-N residents are never
            // evicted; if nothing is evictable the limit is exceeded until
            // the periodic residency cycle can free a slot.
            enforce_capacity(map);
            map.insert(
                k.to_string(),
                EngineInfo {
                    engine: engine.clone(),
                    idle_task: Some(idle_task),
                },
            );
        }
    }

    Ok(engine)
}

/// If the detected language is not supported by any model, fall back to "en"
/// (so auto-detection never produces a language that cannot be translated).
fn supported_or_en(lang: &str) -> String {
    if lang == "en" || crate::models::is_supported_language(lang) {
        lang.to_string()
    } else {
        "en".to_string()
    }
}

/// Convenience alias for single-text translation with automatic pivot routing.
#[inline]
pub async fn translate(
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String, ServiceError> {
    translate_with_pivot(from_lang, to_lang, text).await
}

/// Translate a single text with pivot routing, returning the translated text.
pub async fn translate_with_pivot(
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String, ServiceError> {
    translate_with_pivot_formats(from_lang, to_lang, text, None).await
}

/// Translate a single text with caller-specified parse formats.
pub async fn translate_with_pivot_formats(
    from_lang: &str,
    to_lang: &str,
    text: &str,
    parse_formats: Option<config::ParseFormats>,
) -> Result<String, ServiceError> {
    translate_with_pivot_timed_formats(from_lang, to_lang, text, parse_formats)
        .await
        .map(|out| out.text.into_string())
}

/// Translate a single text with pivot routing, returning both the translated text
/// and the effective source language code.
pub async fn translate_with_pivot_detected(
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<(String, String), ServiceError> {
    translate_with_pivot_detected_formats(from_lang, to_lang, text, None).await
}

/// Translate a single text with caller-specified parse formats, returning both
/// the translated text and the effective source language code.
pub async fn translate_with_pivot_detected_formats(
    from_lang: &str,
    to_lang: &str,
    text: &str,
    parse_formats: Option<config::ParseFormats>,
) -> Result<(String, String), ServiceError> {
    translate_with_pivot_timed_formats(from_lang, to_lang, text, parse_formats)
        .await
        .map(|out| (out.text.into_string(), out.detected_language))
}

/// Translate a single text with pivot routing, measuring translation duration and model load duration.
pub async fn translate_with_pivot_timed(
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<TranslationOutput, ServiceError> {
    translate_with_pivot_timed_formats(from_lang, to_lang, text, None).await
}

/// Translate a single text with caller-specified parse formats, measuring translation duration and model load duration.
pub async fn translate_with_pivot_timed_formats(
    from_lang: &str,
    to_lang: &str,
    text: &str,
    parse_formats: Option<config::ParseFormats>,
) -> Result<TranslationOutput, ServiceError> {
    if text.is_empty() {
        return Ok(TranslationOutput {
            text: TranslatedText::default(),
            detected_language: if from_lang == "auto" {
                String::new()
            } else {
                from_lang.to_string()
            },
            translation_duration: Duration::ZERO,
            model_load_duration: Duration::ZERO,
        });
    }

    if from_lang != "auto" && from_lang == to_lang {
        return Ok(TranslationOutput {
            text: text.into(),
            detected_language: from_lang.to_string(),
            translation_duration: Duration::ZERO,
            model_load_duration: Duration::ZERO,
        });
    }

    let effective_formats = parse_formats.unwrap_or_else(|| {
        if config::is_initialized() {
            config::get().parse_formats
        } else {
            config::ParseFormats::default()
        }
    });
    let max_sent_len = if config::is_initialized() {
        config::get().max_sentence_length
    } else {
        512
    };

    let has_markup = (effective_formats.has_html() && text.contains('<'))
        || (effective_formats.has_md() && crate::engine::has_markdown_syntax(text));
    let is_short = has_markup || !crate::engine::text::exceeds_char_limit(text, max_sent_len);
    if from_lang != "auto" {
        let (engines, model_load_duration) = ensure_required_engines(from_lang, to_lang).await?;
        let (translated, translation_duration) = if is_short {
            let (t, d) = translate_segment_with_engines(&engines, from_lang, to_lang, text, effective_formats).await?;
            (t.normalize_punctuation(to_lang), d)
        } else {
            // Long explicit-language text goes through segmentation.
            let (t, d) = translate_long_text_with_engines(&engines, from_lang, to_lang, text, effective_formats).await?;
            (normalize_output_punctuation(t, to_lang), d)
        };
        Ok(TranslationOutput {
            text: translated,
            detected_language: from_lang.to_string(),
            translation_duration,
            model_load_duration,
        })
    } else {
        // Long text or auto: segment by language detection
        let segments = detector::detect_multiple_languages(text).await;

        if segments.len() <= 1 {
            let effective_from = if segments.len() == 1 {
                supported_or_en(&segments[0].language)
            } else {
                let detected = detector::detect_language(text).await;
                if detected.is_empty() {
                    return Err(ServiceError::LanguageDetectionFailed(
                        "Failed to detect source language".to_string(),
                    ));
                }
                supported_or_en(&detected)
            };

            if effective_from == to_lang {
                return Ok(TranslationOutput {
                    text: text.into(),
                    detected_language: effective_from,
                    translation_duration: Duration::ZERO,
                    model_load_duration: Duration::ZERO,
                });
            }

            let (engines, model_load_duration) = ensure_required_engines(&effective_from, to_lang).await?;

            let is_long = if has_markup {
                false
            } else {
                !is_short
            };
            let (final_text, translation_duration) = if is_long {
                let (t, d) = translate_long_text_with_engines(&engines, &effective_from, to_lang, text, effective_formats).await?;
                (normalize_output_punctuation(t, to_lang), d)
            } else {
                let (t, d) = translate_segment_with_engines(&engines, &effective_from, to_lang, text, effective_formats).await?;
                (t.normalize_punctuation(to_lang), d)
            };

            Ok(TranslationOutput {
                text: final_text,
                detected_language: effective_from,
                translation_duration,
                model_load_duration,
            })
        } else {
            let mut total_load_duration = Duration::ZERO;
            let mut engine_map: HashMap<String, PreparedEngines> = HashMap::new();
            for seg in &segments {
                if seg.language != to_lang {
                    let seg_lang = supported_or_en(&seg.language);
                    #[allow(clippy::map_entry)]
                    if !engine_map.contains_key(&seg_lang) {
                        let (engines, d) = ensure_required_engines(&seg_lang, to_lang).await?;
                        total_load_duration += d;
                        engine_map.insert(seg_lang, engines);
                    }
                }
            }

            let effective_from = supported_or_en(&segments[0].language);
            let mut total_translation_duration = Duration::ZERO;
            let mut out = String::with_capacity(text.len());
            let mut last_end = 0usize;
            for seg in &segments {
                if seg.start > last_end {
                    out.push_str(&text[last_end..seg.start]);
                }
                if seg.language == to_lang {
                    out.push_str(&seg.text);
                } else {
                    let seg_lang = supported_or_en(&seg.language);
                    let translated = if let Some(engines) = engine_map.get(&seg_lang) {
                        match translate_segment_with_engines(engines, &seg_lang, to_lang, &seg.text, effective_formats).await {
                            Ok((t, d)) => {
                                total_translation_duration += d;
                                t
                            }
                            Err(e) => {
                                crate::log_error!("Failed to translate segment: {}", e);
                                TranslatedText::String(seg.text.clone())
                            }
                        }
                    } else {
                        match translate_segment(&seg_lang, to_lang, &seg.text, effective_formats).await {
                            Ok((t, d)) => {
                                total_translation_duration += d;
                                TranslatedText::String(t)
                            }
                            Err(e) => {
                                crate::log_error!("Failed to translate segment: {}", e);
                                TranslatedText::String(seg.text.clone())
                            }
                        }
                    };
                    out.push_str(&translated);
                }
                last_end = seg.end;
            }
            if last_end < text.len() {
                out.push_str(&text[last_end..]);
            }
            let final_text = normalize_output_punctuation(out, to_lang);

            Ok(TranslationOutput {
                text: final_text,
                detected_language: effective_from,
                translation_duration: total_translation_duration,
                model_load_duration: total_load_duration,
            })
        }
    }
}

/// Translate a batch of texts with pivot routing and caller-specified parse formats.
/// Returns a vector of tuples containing `(translated_text, detected_language)` aligned with the input `texts`.
pub async fn translate_batch_with_pivot_formats(
    from_lang: &str,
    to_lang: &str,
    texts: &[String],
    parse_formats: Option<config::ParseFormats>,
) -> Result<Vec<(String, String)>, ServiceError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let max_threads = if config::is_initialized() {
        config::get().max_threads
    } else {
        0
    };
    let concurrency = if max_threads > 0 {
        max_threads
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1)
    };

    if from_lang != "auto" && from_lang == to_lang {
        let out = texts
            .iter()
            .map(|t| (t.clone(), from_lang.to_string()))
            .collect();
        return Ok(out);
    }

    if from_lang != "auto" {
        let effective_formats = parse_formats.unwrap_or_else(|| {
            if config::is_initialized() {
                config::get().parse_formats
            } else {
                config::ParseFormats::default()
            }
        });
        let limit = if config::is_initialized() {
            let cfg = config::get();
            if cfg.max_sentence_length > 0 {
                cfg.max_sentence_length
            } else {
                512
            }
        } else {
            512
        };

        let (engines, _) = ensure_required_engines(from_lang, to_lang).await?;
        let from_tag = from_lang.to_string();
        let futures_iter: Vec<_> = texts
            .iter()
            .map(|text| async {
                translate_batch_item_with_engines(
                    &engines,
                    from_lang,
                    to_lang,
                    text,
                    effective_formats,
                    limit,
                )
                .await
            })
            .collect();

        let results = futures::stream::iter(futures_iter)
            .buffered(concurrency)
            .collect::<Vec<Result<String, ServiceError>>>()
            .await;

        let mut out = Vec::with_capacity(results.len());
        for res in results {
            out.push((res?, from_tag.clone()));
        }
        Ok(out)
    } else {
        let futures_iter: Vec<_> = texts
            .iter()
            .map(|text| translate_batch_item_auto(to_lang, text, parse_formats))
            .collect();
        futures::stream::iter(futures_iter)
            .buffered(concurrency)
            .collect::<Vec<Result<(String, String), ServiceError>>>()
            .await
            .into_iter()
            .collect()
    }
}

async fn translate_batch_item_with_engines(
    engines: &PreparedEngines,
    from_lang: &str,
    to_lang: &str,
    text: &str,
    effective_formats: config::ParseFormats,
    limit: usize,
) -> Result<String, ServiceError> {
    if text.is_empty() {
        return Ok(String::new());
    }

    let has_markup = (effective_formats.has_html() && text.contains('<'))
        || (effective_formats.has_md() && crate::engine::has_markdown_syntax(text));
    let is_short = has_markup || !crate::engine::text::exceeds_char_limit(text, limit);

    let translated = if is_short {
        let (t, _) = translate_segment_with_engines(
            engines,
            from_lang,
            to_lang,
            text,
            effective_formats,
        )
        .await?;
        t.normalize_punctuation(to_lang)
    } else {
        let (t, _) = translate_long_text_with_engines(
            engines,
            from_lang,
            to_lang,
            text,
            effective_formats,
        )
        .await?;
        normalize_output_punctuation(t, to_lang)
    };

    Ok(translated.into_string())
}

async fn translate_batch_item_auto(
    to_lang: &str,
    text: &str,
    parse_formats: Option<config::ParseFormats>,
) -> Result<(String, String), ServiceError> {
    if text.is_empty() {
        return Ok((String::new(), String::new()));
    }
    translate_with_pivot_detected_formats("auto", to_lang, text, parse_formats).await
}

/// Convenience alias for batch translation with automatic pivot routing.
#[inline]
pub async fn translate_batch(
    from_lang: &str,
    to_lang: &str,
    texts: &[String],
) -> Result<Vec<(String, String)>, ServiceError> {
    translate_batch_with_pivot(from_lang, to_lang, texts).await
}

/// Translate a batch of texts with pivot routing.
pub async fn translate_batch_with_pivot(
    from_lang: &str,
    to_lang: &str,
    texts: &[String],
) -> Result<Vec<(String, String)>, ServiceError> {
    translate_batch_with_pivot_formats(from_lang, to_lang, texts, None).await
}

/// Translate a single segment using prepared engines (may use pivot).
async fn translate_segment_with_engines(
    engines: &PreparedEngines,
    from_lang: &str,
    to_lang: &str,
    text: &str,
    effective_formats: config::ParseFormats,
) -> Result<(TranslatedText, Duration), ServiceError> {
    if text.is_empty() {
        return Ok((TranslatedText::default(), Duration::ZERO));
    }

    match engines {
        PreparedEngines::SameLanguage => {
            Ok((TranslatedText::String(text.to_string()), Duration::ZERO))
        }
        PreparedEngines::Direct(engine) => {
            translate_single_language_text_with_engine(engine, from_lang, to_lang, text, effective_formats).await
        }
        PreparedEngines::Pivot(first, second) => {
            let formats_str = effective_formats.as_static_str();
            let rules_version = crate::engine::rules::RuleRegistry::global().version();
            let cache_key = crate::utils::cache::enabled().then(|| {
                crate::utils::cache::sync_rules_version(rules_version);
                crate::utils::cache::compute_key_with_version(
                    &[from_lang, to_lang, text, formats_str],
                    rules_version,
                )
            });
            if let Some(key) = cache_key {
                if let Some(cached) = crate::utils::cache::read_cached(key) {
                    return Ok((TranslatedText::Cached(cached), Duration::ZERO));
                }
            }

            let (intermediate, d1) =
                translate_single_language_text_with_engine(first, from_lang, "en", text, effective_formats).await?;
            let (final_text, d2) =
                translate_single_language_text_with_engine(second, "en", to_lang, &intermediate, effective_formats).await?;
            if let Some(key) = cache_key {
                match &final_text {
                    TranslatedText::Cached(arc) => crate::utils::cache::set(key, arc.clone()),
                    TranslatedText::String(s) => crate::utils::cache::write_cached(s, key),
                }
            }
            Ok((final_text, d1 + d2))
        }
    }
}

/// Translate a single segment (may use pivot).
pub async fn translate_segment(
    from_lang: &str,
    to_lang: &str,
    text: &str,
    effective_formats: config::ParseFormats,
) -> Result<(String, Duration), ServiceError> {
    if text.is_empty() {
        return Ok((String::new(), Duration::ZERO));
    }
    if from_lang == to_lang {
        return Ok((text.to_string(), Duration::ZERO));
    }
    let (engines, _) = ensure_required_engines(from_lang, to_lang).await?;
    let (translated, duration) =
        translate_segment_with_engines(&engines, from_lang, to_lang, text, effective_formats).await?;
    Ok((translated.into_string(), duration))
}

/// Translate with a single directly-held engine.
async fn translate_single_language_text_with_engine(
    engine: &Arc<TranslationEngine>,
    from_lang: &str,
    to_lang: &str,
    text: &str,
    effective_formats: config::ParseFormats,
) -> Result<(TranslatedText, Duration), ServiceError> {
    engine.record_call();

    // Cache key incorporates effective_formats and language rules version to prevent cache collisions
    // between different parse modes or stale cache when rules are updated/reloaded.
    let formats_str = effective_formats.as_static_str();
    let rules_version = crate::engine::rules::RuleRegistry::global().version();
    let cache_key = crate::utils::cache::enabled().then(|| {
        crate::utils::cache::sync_rules_version(rules_version);
        crate::utils::cache::compute_key_with_version(
            &[from_lang, to_lang, text, formats_str],
            rules_version,
        )
    });
    if let Some(key) = cache_key {
        if let Some(cached) = crate::utils::cache::read_cached(key) {
            return Ok((TranslatedText::Cached(cached), Duration::ZERO));
        }
    }

    let _permit = engine_gate()
        .acquire()
        .await
        .map_err(|e| ServiceError::Internal(format!("Engine gate closed: {}", e)))?;
    let engine_ref = engine.clone();
    let text_owned = text.to_string();
    let (result, duration) = tokio::task::spawn_blocking(move || {
        let t0 = Instant::now();
        let res = engine_ref.translate_with_formats(&text_owned, effective_formats);
        let dur = t0.elapsed();
        (res, dur)
    })
    .await
    .map_err(|e| {
        ServiceError::Internal(format!("Translation worker panicked: {}", e))
    })?;
    if let Some(key) = cache_key {
        crate::utils::cache::write_cached(&result, key);
    }
    Ok((TranslatedText::String(result), duration))
}


/// Translate a long text by splitting into sentences using prepared engines.
pub async fn translate_long_text_with_engines(
    engines: &PreparedEngines,
    from_lang: &str,
    to_lang: &str,
    text: &str,
    effective_formats: config::ParseFormats,
) -> Result<(String, Duration), ServiceError> {
    if from_lang == to_lang {
        return Ok((text.to_string(), Duration::ZERO));
    }
    let sentences = crate::engine::markup::split::split_sentences_bounded_with_lang(
        text,
        crate::engine::markup::split::MAX_SENTENCE_LEN,
        Some(from_lang),
    );

    if sentences.is_empty() {
        return Ok((String::new(), Duration::ZERO));
    }

    if sentences.len() == 1 {
        let sent = &sentences[0];
        if sent.is_blank() {
            let mut out = sent.text.to_string();
            out.push_str(&sent.separator);
            return Ok((out, Duration::ZERO));
        }
        let (t, d) = translate_segment_with_engines(
            engines,
            from_lang,
            to_lang,
            &sent.text,
            effective_formats,
        )
        .await?;
        let mut out = t.into_string();
        crate::utils::lang::append_split_sentence_separator(&mut out, &sent.separator);
        return Ok((out, d));
    }

    let start_wall = Instant::now();
    let num_sentences = sentences.len();
    let mut results: Vec<Option<TranslatedText>> = (0..num_sentences).map(|_| None).collect();
    let mut uncached_indices: Vec<usize> = Vec::new();

    let formats_str = effective_formats.as_static_str();
    let rules_version = crate::engine::rules::RuleRegistry::global().version();
    let cache_enabled = crate::utils::cache::enabled();
    if cache_enabled {
        crate::utils::cache::sync_rules_version(rules_version);
    }

    // 1. Synchronous check: blank sentences and cache hits without any async permit overhead
    for (i, sent) in sentences.iter().enumerate() {
        if sent.is_blank() {
            results[i] = Some(TranslatedText::String(sent.text.to_string()));
            continue;
        }

        if cache_enabled {
            let key = crate::utils::cache::compute_key_with_version(
                &[from_lang, to_lang, &sent.text, formats_str],
                rules_version,
            );
            if let Some(cached) = crate::utils::cache::read_cached(key) {
                results[i] = Some(TranslatedText::Cached(cached));
                continue;
            }
        }

        uncached_indices.push(i);
    }

    // 2. If there are uncached sentences, translate them
    if !uncached_indices.is_empty() {
        if uncached_indices.len() == 1 {
            let idx = uncached_indices[0];
            let (t, _) = translate_segment_with_engines(
                engines,
                from_lang,
                to_lang,
                &sentences[idx].text,
                effective_formats,
            )
            .await?;
            results[idx] = Some(t);
        } else {
            let max_threads = if crate::config::is_initialized() {
                crate::config::get().max_threads
            } else {
                0
            };
            let concurrency = if max_threads > 0 {
                max_threads
            } else {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .max(1)
            };
            let text_items: Vec<(usize, String)> = uncached_indices
                .iter()
                .map(|&idx| (idx, sentences[idx].text.to_string()))
                .collect();

            use futures::StreamExt;
            let translated_items = futures::stream::iter(text_items)
                .map(|(idx, sent_text)| async move {
                    let res = translate_segment_with_engines(
                        engines,
                        from_lang,
                        to_lang,
                        &sent_text,
                        effective_formats,
                    )
                    .await;
                    (idx, res)
                })
                .buffered(concurrency)
                .collect::<Vec<_>>()
                .await;

            for (idx, res) in translated_items {
                let (t, _) = res?;
                results[idx] = Some(t);
            }
        }
    }

    let total_duration = start_wall.elapsed();
    let mut out = String::with_capacity(text.len());

    for (i, (res_opt, sent)) in results.into_iter().zip(&sentences).enumerate() {
        let piece = match res_opt {
            Some(t) => t,
            None => TranslatedText::String(sent.text.to_string()),
        };

        if i > 0 && crate::utils::lang::should_insert_adaptive_space(&out, piece.as_str(), Some(to_lang)) {
            out.push(' ');
        }

        out.push_str(piece.as_str());
        crate::utils::lang::append_split_sentence_separator(&mut out, &sent.separator);
    }

    Ok((out, total_duration))
}

/// Translate a long text by splitting into sentences.
pub async fn translate_long_text(
    from_lang: &str,
    to_lang: &str,
    text: &str,
    effective_formats: config::ParseFormats,
) -> Result<(String, Duration), ServiceError> {
    let (engines, _) = ensure_required_engines(from_lang, to_lang).await?;
    translate_long_text_with_engines(&engines, from_lang, to_lang, text, effective_formats).await
}

/// Format Chinese punctuation for `zh` targets: map ASCII punctuation to its
/// fullwidth counterpart (`,`→`，`, `?`→`？`, `!`→`！`, `;`→`；`, `:`→`：`).
///
/// The base translation models emit half-width `,/?/:` (and occasionally
/// `;!`) for Chinese output; this normalises them to the fullwidth forms the
/// Chinese typography expects. A single ASCII space immediately following the
/// punctuation is consumed together with it ("`, `" → "`，`"), matching the
/// English "comma + space" style to the Chinese "no space" style; extra
/// spaces beyond the first are preserved.
/// HTML tags (`<...>`, including attributes) and HTML entities (`&...;`) are
/// skipped verbatim so markup survives untouched.
/// Returns a borrowed slice of `text` when no conversion applies (non-zh
/// targets, HTML, feature off, or no convertible punctuation) and an owned
/// converted copy otherwise.
/// Apply Chinese punctuation normalization and Pangu spacing (盘古之白).
pub fn format_zh_punctuation<'a>(
    text: &'a str,
    to_lang: &str,
) -> std::borrow::Cow<'a, str> {
    // Fast path: nothing to format for non-Chinese targets
    if !to_lang.starts_with("zh") {
        return std::borrow::Cow::Borrowed(text);
    }
    apply_zh_punct_and_pangu(std::borrow::Cow::Borrowed(text))
}

/// Apply fullwidth Chinese punctuation normalization with context-sensitive guards
/// for numeric formatting (thousands separators, European decimals), colons in timestamps,
/// URLs, host:port, and Rust double colons, as well as HTML tags and Markdown code fences.
#[inline]
#[allow(dead_code)]
pub(crate) fn format_zh_punct_marks<'a>(text: &'a str) -> std::borrow::Cow<'a, str> {
    apply_zh_punct_marks(std::borrow::Cow::Borrowed(text))
}

/// Core punctuation normalization operating on `Cow<'a, str>` to preserve zero allocations.
pub(crate) fn apply_zh_punct_marks<'a>(text: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'a, str> {
    let s = text.as_ref();
    // Fast path: no convertible ASCII punctuation -> no allocation needed.
    let has_convertible = s
        .bytes()
        .any(|b| matches!(b, b',' | b'?' | b'!' | b';' | b':'))
        || (s.as_bytes().iter().any(|&b| b >= 0x80) && s.contains(' '));
    if !has_convertible {
        return text;
    }
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return text;
    }
    let mut out: Option<String> = None;
    let mut last = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        // HTML tags (`<...>`) and entities (`&...;`) are kept verbatim: their
        // punctuation/attributes must not be fullwidth-converted.
        if b == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                i = tag_end;
                continue;
            }
        } else if b == b'&' {
            if let Some(ent_end) = html_entity_end(bytes, i) {
                i = ent_end;
                continue;
            }
        } else if b == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                i = code_end;
                continue;
            }
        } else if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(v_end) = shell_var_end(bytes, i) {
                i = v_end;
                continue;
            }
        } else if b == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(t_end) = template_var_end(bytes, i) {
                i = t_end;
                continue;
            }
        } else if let Some(u_end) = url_end(bytes, i) {
            i = u_end;
            continue;
        }

        let replacement = match b {
            b',' => {
                let digit_before = i > 0 && bytes[i - 1].is_ascii_digit();
                let digit_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
                if digit_before && digit_after {
                    None
                } else {
                    Some(('，', next_after_punct_byte(bytes, i)))
                }
            }
            b'?' => {
                let dot_after = i + 1 < bytes.len() && bytes[i + 1] == b'.';
                let qmark_after = i + 1 < bytes.len() && bytes[i + 1] == b'?';
                let bracket_after = i + 1 < bytes.len() && bytes[i + 1] == b'[';
                let qmark_before = i > 0 && bytes[i - 1] == b'?';
                if dot_after || qmark_after || bracket_after || qmark_before {
                    None
                } else {
                    Some(('？', next_after_punct_byte(bytes, i)))
                }
            }
            b'!' => {
                let eq_after = i + 1 < bytes.len() && bytes[i + 1] == b'=';
                let excl_after = i + 1 < bytes.len() && bytes[i + 1] == b'!';
                let excl_before = i > 0 && bytes[i - 1] == b'!';
                let macro_call = i > 0
                    && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')
                    && i + 1 < bytes.len()
                    && matches!(bytes[i + 1], b'(' | b'[' | b'{');
                let ident_after = i + 1 < bytes.len()
                    && (bytes[i + 1].is_ascii_alphanumeric() || bytes[i + 1] == b'_');
                let ident_op = ident_after
                    && (i == 0
                        || bytes[i - 1].is_ascii_whitespace()
                        || matches!(
                            bytes[i - 1],
                            b'(' | b'[' | b'{' | b'!' | b'&' | b'|' | b'+' | b'-' | b'*' | b'/' | b'='
                        ));
                if eq_after || excl_after || excl_before || macro_call || ident_op {
                    None
                } else {
                    Some(('！', next_after_punct_byte(bytes, i)))
                }
            }
            b';' => Some(('；', next_after_punct_byte(bytes, i))),
            b':' => {
                let colon_before = i > 0 && bytes[i - 1] == b':';
                let colon_after = i + 1 < bytes.len() && bytes[i + 1] == b':';
                let scheme_after =
                    i + 2 < bytes.len() && bytes[i + 1] == b'/' && bytes[i + 2] == b'/';
                let alnum_before = i > 0 && bytes[i - 1].is_ascii_alphanumeric();
                let digit_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
                let hex_before = i > 0 && bytes[i - 1].is_ascii_hexdigit();
                let hex_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_hexdigit();
                let bracket_port = is_bracket_port(bytes, i);
                let line_header = is_line_head_header(bytes, i);
                let quote_key_before = i > 0 && (bytes[i - 1] == b'"' || bytes[i - 1] == b'\'');
                let op_after = i + 1 < bytes.len()
                    && matches!(bytes[i + 1], b'-' | b'=' | b'+' | b'?' | b'/' | b'\\');
                let symbol_before = i > 0
                    && matches!(bytes[i - 1], b'-' | b'=' | b'+' | b'?' | b'$' | b'%' | b'@' | b'{');
                if colon_before
                    || colon_after
                    || scheme_after
                    || (alnum_before && digit_after)
                    || (hex_before && hex_after)
                    || bracket_port
                    || line_header
                    || quote_key_before
                    || op_after
                    || symbol_before
                {
                    None
                } else {
                    Some(('：', next_after_punct_byte(bytes, i)))
                }
            }
            _ => None,
        };

        if let Some((fullwidth, next_i)) = replacement {
            let buf = out.get_or_insert_with(|| {
                let mut s = String::with_capacity(text.len() + 16);
                s.push_str(&text[..last]);
                s
            });
            if i > last {
                buf.push_str(&text[last..i]);
            }
            buf.push(fullwidth);
            i = next_i;
            last = next_i;
        } else if b >= 0x80 && s.is_char_boundary(i) {
            if let Some(curr) = s[i..].chars().next() {
                let curr_len = curr.len_utf8();
                let curr_end = i + curr_len;
                if crate::utils::lang::is_cjk_fullwidth_punct(curr) && bytes.get(curr_end) == Some(&b' ') {
                    let buf = out.get_or_insert_with(|| {
                        let mut s_buf = String::with_capacity(text.len() + 16);
                        s_buf.push_str(&text[..last]);
                        s_buf
                    });
                    if i > last {
                        buf.push_str(&text[last..i]);
                    }
                    buf.push(curr);
                    i = curr_end + 1;
                    last = i;
                    continue;
                }
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    match out {
        None => text,
        Some(mut buf) => {
            if last < s.len() {
                buf.push_str(&s[last..]);
            }
            std::borrow::Cow::Owned(buf)
        }
    }
}

pub(crate) use crate::utils::lang::is_cjk_char;

/// End index of URL starting at `start` if `bytes[start..]` begins with a known URL scheme.
#[inline]
fn url_end(bytes: &[u8], start: usize) -> Option<usize> {
    let schemes: [&[u8]; 4] = [b"http://", b"https://", b"ftp://", b"file://"];
    let is_scheme = schemes.iter().any(|s| bytes[start..].starts_with(s));
    if !is_scheme {
        return None;
    }
    let mut end = start;
    while end < bytes.len() {
        let b = bytes[end];
        if b <= b' ' || b >= 127 || matches!(b, b'<' | b'>' | b'"' | b'\'' | b'`') {
            break;
        }
        end += 1;
    }
    while end > start && matches!(bytes[end - 1], b'.' | b',' | b';' | b'!' | b'?') {
        end -= 1;
    }
    if end > start {
        Some(end)
    } else {
        None
    }
}

/// Checks if colon at `i` is immediately preceded by `]` and followed by digits,
/// representing an IPv6 network port (e.g. `[2001:db8::1]:8080`).
#[inline]
fn is_bracket_port(bytes: &[u8], i: usize) -> bool {
    i > 0 && bytes[i - 1] == b']' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()
}

/// Checks if colon at `colon_idx` is preceded by a line-initial `[A-Za-z0-9_-]+` token
/// (e.g. `Content-Type:`, `Host:`, `auth_token:`), indicating a pseudo-header that
fn is_line_head_header(bytes: &[u8], colon_idx: usize) -> bool {
    if colon_idx == 0 {
        return false;
    }
    if colon_idx + 1 < bytes.len() {
        let after = bytes[colon_idx + 1];
        if after != b' ' && after != b'\t' && after != b'\n' && after != b'\r' {
            return false;
        }
    }
    let mut p = colon_idx;
    while p > 0 {
        let b = bytes[p - 1];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
            p -= 1;
        } else {
            break;
        }
    }
    if p == colon_idx {
        return false;
    }
    let mut line_start = p;
    while line_start > 0 && (bytes[line_start - 1] == b' ' || bytes[line_start - 1] == b'\t') {
        line_start -= 1;
    }
    line_start == 0 || bytes[line_start - 1] == b'\n' || bytes[line_start - 1] == b'\r'
}

/// Checks if an ASCII digit at `i` is followed by a known unit (e.g. MB, GB, KB, TB, PB)
/// that should be spaced (e.g. "100MB" -> "100 MB").
#[inline]
fn is_digit_unit_boundary(bytes: &[u8], i: usize) -> bool {
    if !bytes[i].is_ascii_digit() || i + 1 >= bytes.len() {
        return false;
    }
    let rest = &bytes[i + 1..];
    let units: [&[u8]; 9] = [b"MB", b"GB", b"KB", b"TB", b"PB", b"EB", b"Gbps", b"Mbps", b"Kbps"];
    for unit in units {
        if rest.starts_with(unit) {
            let after_unit = i + 1 + unit.len();
            if after_unit == bytes.len() || !bytes[after_unit].is_ascii_alphanumeric() {
                return true;
            }
        }
    }
    false
}

/// Apply Pangu Spacing (盘古之白): inserts a halfwidth space at boundaries between
/// CJK characters and ASCII letters/digits, as well as between digits and units (e.g. 100MB -> 100 MB),
/// while strictly protecting HTML tags, Markdown code blocks/spans, URLs, and existing fullwidth punctuation.
pub(crate) fn apply_pangu_spacing<'a>(text: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'a, str> {
    let s = text.as_ref();
    if s.is_empty() {
        return text;
    }
    let bytes = s.as_bytes();
    let mut out: Option<String> = None;
    let mut last_copied = 0;
    let mut i = 0;
    let mut prev_char: Option<char> = None;
    let mut prev_end = 0;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                prev_char = None;
                i = tag_end;
                prev_end = i;
                continue;
            }
        } else if bytes[i] == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                if let Some(prev) = prev_char {
                    if is_cjk_char(prev) {
                        if out.is_none() {
                            out = Some(String::with_capacity(s.len() + 16));
                        }
                        if let Some(buf) = &mut out {
                            buf.push_str(&s[last_copied..i]);
                            buf.push(' ');
                            last_copied = i;
                        }
                    }
                }
                if let Some(next_c) = s[code_end..].chars().next() {
                    if is_cjk_char(next_c) {
                        if out.is_none() {
                            out = Some(String::with_capacity(s.len() + 16));
                        }
                        if let Some(buf) = &mut out {
                            buf.push_str(&s[last_copied..code_end]);
                            buf.push(' ');
                            last_copied = code_end;
                        }
                    }
                }
                prev_char = None;
                i = code_end;
                prev_end = i;
                continue;
            }
        } else if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(v_end) = shell_var_end(bytes, i) {
                prev_char = None;
                i = v_end;
                prev_end = i;
                continue;
            }
        } else if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(t_end) = template_var_end(bytes, i) {
                prev_char = None;
                i = t_end;
                prev_end = i;
                continue;
            }
        } else if let Some(u_end) = url_end(bytes, i) {
            if let Some(prev) = prev_char {
                if is_cjk_char(prev) {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    if let Some(buf) = &mut out {
                        buf.push_str(&s[last_copied..i]);
                        buf.push(' ');
                        last_copied = i;
                    }
                }
            }
            let last_b = bytes[u_end - 1];
            prev_char = if last_b.is_ascii_alphanumeric() {
                Some(last_b as char)
            } else {
                None
            };
            i = u_end;
            prev_end = i;
            continue;
        }

        if !s.is_char_boundary(i) {
            i += 1;
            continue;
        }
        let Some(curr) = s[i..].chars().next() else {
            break;
        };
        let curr_len = curr.len_utf8();
        let curr_end = i + curr_len;

        if let Some(prev) = prev_char {
            let should_insert = (is_cjk_char(prev) && curr.is_ascii_alphanumeric())
                || (prev.is_ascii_alphanumeric() && is_cjk_char(curr))
                || (prev.is_ascii_digit() && is_digit_unit_boundary(bytes, prev_end - 1));

            if should_insert {
                if out.is_none() {
                    out = Some(String::with_capacity(s.len() + 16));
                }
                if let Some(buf) = &mut out {
                    buf.push_str(&s[last_copied..i]);
                    buf.push(' ');
                    last_copied = i;
                }
            }
        }

        if curr.is_whitespace() || (!curr.is_ascii_alphanumeric() && !is_cjk_char(curr)) {
            prev_char = None;
        } else {
            prev_char = Some(curr);
        }
        prev_end = curr_end;
        i = curr_end;
    }

    match out {
        None => text,
        Some(mut buf) => {
            if last_copied < s.len() {
                buf.push_str(&s[last_copied..]);
            }
            std::borrow::Cow::Owned(buf)
        }
    }
}

/// Convenience wrapper around `apply_pangu_spacing` for borrowed strings.
#[inline]
pub fn format_pangu_spacing<'a>(text: &'a str) -> std::borrow::Cow<'a, str> {
    apply_pangu_spacing(std::borrow::Cow::Borrowed(text))
}

/// Fused single-pass Chinese punctuation normalization and Pangu spacing.
/// Minimizes heap allocations and string scanning overhead in the hot translation post-processing path.
pub(crate) fn apply_zh_punct_and_pangu<'a>(
    text: std::borrow::Cow<'a, str>,
) -> std::borrow::Cow<'a, str> {
    let s = text.as_ref();
    if s.is_empty() {
        return text;
    }
    let bytes = s.as_bytes();
    let mut has_convertible_punct = false;
    let mut has_cjk_boundary_candidate = false;
    for &b in bytes {
        if matches!(b, b',' | b'?' | b'!' | b';' | b':') {
            has_convertible_punct = true;
        }
        if b >= 0x80 {
            has_cjk_boundary_candidate = true;
        }
        if has_convertible_punct && has_cjk_boundary_candidate {
            break;
        }
    }

    if !has_convertible_punct && !has_cjk_boundary_candidate {
        return text;
    }
    let mut out: Option<String> = None;
    let mut last_copied = 0;
    let mut i = 0;
    let mut prev_char: Option<char> = None;
    let mut prev_end = 0;

    while i < bytes.len() {
        let b = bytes[i];

        if b == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                prev_char = None;
                i = tag_end;
                prev_end = i;
                continue;
            }
        } else if b == b'&' {
            if let Some(ent_end) = html_entity_end(bytes, i) {
                prev_char = None;
                i = ent_end;
                prev_end = i;
                continue;
            }
        } else if b == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                if let Some(prev) = prev_char {
                    if is_cjk_char(prev) {
                        let buf = out.get_or_insert_with(|| {
                            let mut s_buf = String::with_capacity(s.len() + 16);
                            s_buf.push_str(&s[..last_copied]);
                            s_buf
                        });
                        buf.push_str(&s[last_copied..i]);
                        buf.push(' ');
                        last_copied = i;
                    }
                }
                if let Some(next_c) = s[code_end..].chars().next() {
                    if is_cjk_char(next_c) {
                        let buf = out.get_or_insert_with(|| {
                            let mut s_buf = String::with_capacity(s.len() + 16);
                            s_buf.push_str(&s[..last_copied]);
                            s_buf
                        });
                        buf.push_str(&s[last_copied..code_end]);
                        buf.push(' ');
                        last_copied = code_end;
                    }
                }
                prev_char = None;
                i = code_end;
                prev_end = i;
                continue;
            }
        } else if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(v_end) = shell_var_end(bytes, i) {
                prev_char = None;
                i = v_end;
                prev_end = i;
                continue;
            }
        } else if b == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(t_end) = template_var_end(bytes, i) {
                prev_char = None;
                i = t_end;
                prev_end = i;
                continue;
            }
        } else if let Some(u_end) = url_end(bytes, i) {
            if let Some(prev) = prev_char {
                if is_cjk_char(prev) {
                    let buf = out.get_or_insert_with(|| {
                        let mut s_buf = String::with_capacity(s.len() + 16);
                        s_buf.push_str(&s[..last_copied]);
                        s_buf
                    });
                    buf.push_str(&s[last_copied..i]);
                    buf.push(' ');
                    last_copied = i;
                }
            }
            let last_b = bytes[u_end - 1];
            prev_char = if last_b.is_ascii_alphanumeric() {
                Some(last_b as char)
            } else {
                None
            };
            i = u_end;
            prev_end = i;
            continue;
        }

        let replacement = match b {
            b',' => {
                let digit_before = i > 0 && bytes[i - 1].is_ascii_digit();
                let digit_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
                if digit_before && digit_after {
                    None
                } else {
                    Some(('，', next_after_punct_byte(bytes, i)))
                }
            }
            b'?' => {
                let dot_after = i + 1 < bytes.len() && bytes[i + 1] == b'.';
                let qmark_after = i + 1 < bytes.len() && bytes[i + 1] == b'?';
                let bracket_after = i + 1 < bytes.len() && bytes[i + 1] == b'[';
                let qmark_before = i > 0 && bytes[i - 1] == b'?';
                if dot_after || qmark_after || bracket_after || qmark_before {
                    None
                } else {
                    Some(('？', next_after_punct_byte(bytes, i)))
                }
            }
            b'!' => {
                let eq_after = i + 1 < bytes.len() && bytes[i + 1] == b'=';
                let excl_after = i + 1 < bytes.len() && bytes[i + 1] == b'!';
                let excl_before = i > 0 && bytes[i - 1] == b'!';
                let macro_call = i > 0
                    && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')
                    && i + 1 < bytes.len()
                    && matches!(bytes[i + 1], b'(' | b'[' | b'{');
                let ident_after = i + 1 < bytes.len()
                    && (bytes[i + 1].is_ascii_alphanumeric() || bytes[i + 1] == b'_');
                let ident_op = ident_after
                    && (i == 0
                        || bytes[i - 1].is_ascii_whitespace()
                        || matches!(
                            bytes[i - 1],
                            b'(' | b'[' | b'{' | b'!' | b'&' | b'|' | b'+' | b'-' | b'*' | b'/' | b'='
                        ));
                if eq_after || excl_after || excl_before || macro_call || ident_op {
                    None
                } else {
                    Some(('！', next_after_punct_byte(bytes, i)))
                }
            }
            b';' => Some(('；', next_after_punct_byte(bytes, i))),
            b':' => {
                let colon_before = i > 0 && bytes[i - 1] == b':';
                let colon_after = i + 1 < bytes.len() && bytes[i + 1] == b':';
                let scheme_after =
                    i + 2 < bytes.len() && bytes[i + 1] == b'/' && bytes[i + 2] == b'/';
                let alnum_before = i > 0 && bytes[i - 1].is_ascii_alphanumeric();
                let digit_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
                let hex_before = i > 0 && bytes[i - 1].is_ascii_hexdigit();
                let hex_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_hexdigit();
                let bracket_port = is_bracket_port(bytes, i);
                let line_header = is_line_head_header(bytes, i);
                let quote_key_before = i > 0 && (bytes[i - 1] == b'"' || bytes[i - 1] == b'\'');
                let op_after = i + 1 < bytes.len()
                    && matches!(bytes[i + 1], b'-' | b'=' | b'+' | b'?' | b'/' | b'\\');
                let symbol_before = i > 0
                    && matches!(bytes[i - 1], b'-' | b'=' | b'+' | b'?' | b'$' | b'%' | b'@' | b'{');
                if colon_before
                    || colon_after
                    || scheme_after
                    || (alnum_before && digit_after)
                    || (hex_before && hex_after)
                    || bracket_port
                    || line_header
                    || quote_key_before
                    || op_after
                    || symbol_before
                {
                    None
                } else {
                    Some(('：', next_after_punct_byte(bytes, i)))
                }
            }
            _ => None,
        };

        if let Some((fullwidth, next_i)) = replacement {
            let buf = out.get_or_insert_with(|| {
                let mut s_buf = String::with_capacity(s.len() + 16);
                s_buf.push_str(&s[..last_copied]);
                s_buf
            });
            if i > last_copied {
                buf.push_str(&s[last_copied..i]);
            }
            buf.push(fullwidth);
            prev_char = None;
            i = next_i;
            last_copied = next_i;
            prev_end = next_i;
            continue;
        }

        if !s.is_char_boundary(i) {
            i += 1;
            continue;
        }
        let Some(curr) = s[i..].chars().next() else {
            break;
        };
        let curr_len = curr.len_utf8();
        let curr_end = i + curr_len;

        if crate::utils::lang::is_cjk_fullwidth_punct(curr) && bytes.get(curr_end) == Some(&b' ') {
            let buf = out.get_or_insert_with(|| {
                let mut s_buf = String::with_capacity(s.len() + 16);
                s_buf.push_str(&s[..last_copied]);
                s_buf
            });
            if i > last_copied {
                buf.push_str(&s[last_copied..i]);
            }
            buf.push(curr);
            prev_char = None;
            i = curr_end + 1;
            last_copied = i;
            prev_end = i;
            continue;
        }

        if let Some(prev) = prev_char {
            let should_insert = (is_cjk_char(prev) && curr.is_ascii_alphanumeric())
                || (prev.is_ascii_alphanumeric() && is_cjk_char(curr))
                || (prev.is_ascii_digit() && is_digit_unit_boundary(bytes, prev_end - 1));

            if should_insert {
                let buf = out.get_or_insert_with(|| {
                    let mut s_buf = String::with_capacity(s.len() + 16);
                    s_buf.push_str(&s[..last_copied]);
                    s_buf
                });
                if i > last_copied {
                    buf.push_str(&s[last_copied..i]);
                }
                buf.push(' ');
                last_copied = i;
            }
        }

        if curr.is_whitespace() || (!curr.is_ascii_alphanumeric() && !is_cjk_char(curr)) {
            prev_char = None;
        } else {
            prev_char = Some(curr);
        }
        prev_end = curr_end;
        i = curr_end;
    }

    match out {
        None => text,
        Some(mut buf) => {
            if last_copied < s.len() {
                buf.push_str(&s[last_copied..]);
            }
            std::borrow::Cow::Owned(buf)
        }
    }
}

/// Western Truecasing: For non-Chinese target languages, automatically capitalizes lowercase
/// letters at sentence starts (start of text or after terminal punctuation [.!?] followed by whitespace).
pub fn truecase_western<'a>(text: &'a str, to_lang: &str) -> std::borrow::Cow<'a, str> {
    apply_truecase_western(std::borrow::Cow::Borrowed(text), to_lang)
}

#[inline]
fn is_truecase_nonbreaking_abbr(s: &str, w_start: usize, tok: &str) -> bool {
    let lower = tok.trim_matches('.').to_ascii_lowercase();
    match lower.as_str() {
        "vs" | "v" | "e.g" | "eg" | "i.e" | "ie" | "etc" | "approx" | "ca" | "viz" | "al"
        | "usw" | "bzw" | "d.h" | "z.b" | "p.ex" => true,
        "in" | "ft" | "yd" | "cm" | "mm" | "kg" | "oz" | "lb" | "lbs" | "sec" | "min" | "hr"
        | "hrs" | "no" => {
            let before = s[..w_start].trim_end();
            before.chars().next_back().is_some_and(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}

/// Core Western truecasing operating on `Cow<'a, str>` to preserve zero allocations.
pub fn apply_truecase_western<'a>(text: std::borrow::Cow<'a, str>, to_lang: &str) -> std::borrow::Cow<'a, str> {
    let s = text.as_ref();
    if crate::utils::lang::is_cjk_code(to_lang) || s.is_empty() {
        return text;
    }
    let profile = crate::engine::rules::RuleRegistry::global().get(to_lang);
    let bytes = s.as_bytes();
    let mut out: Option<String> = None;
    let mut last_copied = 0;
    let mut at_sentence_start = true;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                i = tag_end;
                continue;
            }
        } else if bytes[i] == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                i = code_end;
                at_sentence_start = false;
                continue;
            }
        } else if let Some(u_end) = url_end(bytes, i) {
            i = u_end;
            at_sentence_start = false;
            continue;
        }

        if !s.is_char_boundary(i) {
            i += 1;
            continue;
        }
        let Some(c) = s[i..].chars().next() else {
            break;
        };
        let c_len = c.len_utf8();

        if at_sentence_start {
            if c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | '(' | '[' | '{' | '“' | '‘' | '«' | '‹' | '¿' | '¡' | '—' | '–'
                )
            {
                i += c_len;
                continue;
            } else if c.is_lowercase() {
                if out.is_none() {
                    out = Some(String::with_capacity(s.len()));
                }
                if let Some(s) = &mut out {
                    s.push_str(&text[last_copied..i]);
                    for uc in c.to_uppercase() {
                        s.push(uc);
                    }
                    last_copied = i + c_len;
                }
                at_sentence_start = false;
                i += c_len;
                continue;
            } else {
                at_sentence_start = false;
                i += c_len;
                continue;
            }
        }

        if matches!(c, '.' | '!' | '?') {
            let mut peek = i + c_len;
            while peek < bytes.len()
                && matches!(bytes[peek], b'.' | b'!' | b'?' | b'"' | b'\'' | b')' | b']' | b'}')
            {
                peek += 1;
            }
            if peek < bytes.len()
                && (bytes[peek] == b' '
                    || bytes[peek] == b'\t'
                    || bytes[peek] == b'\n'
                    || bytes[peek] == b'\r')
            {
                let mut should_trigger = true;
                if c == '.' {
                    let mut w_start = i;
                    while w_start > 0 {
                        let prev_c = s[..w_start].chars().next_back().unwrap();
                        if prev_c.is_alphanumeric() || prev_c == '.' || prev_c == '-' || prev_c == '_' {
                            w_start -= prev_c.len_utf8();
                        } else {
                            break;
                        }
                    }
                    let tok = &s[w_start..i];
                    let is_abbr = if let Some(prof) = profile.as_ref() {
                        prof.is_nonbreaking_prefix(s, i, w_start) || is_truecase_nonbreaking_abbr(s, w_start, tok)
                    } else {
                        is_truecase_nonbreaking_abbr(s, w_start, tok)
                    };
                    if is_abbr {
                        let rem = s[peek..].trim_start();
                        if let Some(next_c) = rem.chars().next() {
                            if next_c.is_lowercase() {
                                should_trigger = false;
                            }
                        }
                    }
                }
                if should_trigger {
                    at_sentence_start = true;
                }
            }
        }
        if matches!(c, '¿' | '¡') {
            at_sentence_start = true;
        }

        i += c_len;
    }

    match out {
        None => text,
        Some(mut buf) => {
            if last_copied < s.len() {
                buf.push_str(&s[last_copied..]);
            }
            std::borrow::Cow::Owned(buf)
        }
    }
}

/// End index (one past the closing backticks) of the Markdown inline code or code block starting at `start`,
/// where `bytes[start] == b'`'`.
#[inline]
fn markdown_code_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut fence_len = 0;
    while start + fence_len < bytes.len() && bytes[start + fence_len] == b'`' {
        fence_len += 1;
    }
    let fence = &bytes[start..start + fence_len];
    let rest = &bytes[start + fence_len..];
    let mut idx = 0;
    while idx + fence_len <= rest.len() {
        if &rest[idx..idx + fence_len] == fence {
            let is_pre = idx > 0 && rest[idx - 1] == b'`';
            let is_post = idx + fence_len < rest.len() && rest[idx + fence_len] == b'`';
            if !is_pre && !is_post {
                return Some(start + fence_len + idx + fence_len);
            }
        }
        idx += 1;
    }
    None
}

/// End index (one past the closing `>`) of the HTML tag starting at `start`,
/// where `bytes[start] == b'<'`. Returns `None` when there is no closing `>`
/// before the end of the slice.
#[inline]
fn html_tag_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start + 1 >= bytes.len() {
        return None;
    }
    let first_ch = bytes[start + 1];
    if !matches!(first_ch, b'a'..=b'z' | b'A'..=b'Z' | b'/' | b'!' | b'?') {
        return None;
    }
    bytes[start + 1..].iter().position(|&b| b == b'>').map(|off| start + 1 + off + 1)
}

/// End index (one past the `;`) of the HTML entity starting at `start`, where
/// `bytes[start] == b'&'`. Accepted shape: `&` + 1..=31 alphanumeric/`#`
/// characters + `;`. Returns `None` when the candidate is not an entity.
#[inline]
fn html_entity_end(bytes: &[u8], start: usize) -> Option<usize> {
    let rest = &bytes[start + 1..];
    let mut len = 0;
    while len < rest.len() && len < 31 && (rest[len].is_ascii_alphanumeric() || rest[len] == b'#')
    {
        len += 1;
    }
    if len > 0 && len < rest.len() && rest[len] == b';' {
        Some(start + 1 + len + 1)
    } else {
        None
    }
}

/// End index (one past the closing `}`) of shell parameter expansion `${...}` starting at `start`.
#[inline]
fn shell_var_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start + 2 > bytes.len() || bytes[start] != b'$' || bytes[start + 1] != b'{' {
        return None;
    }
    let mut depth = 1;
    let mut idx = start + 2;
    while idx < bytes.len() {
        match bytes[idx] {
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx + 1);
                }
            }
            b'{' => depth += 1,
            b'\n' | b'\r' => return None,
            _ => {}
        }
        idx += 1;
    }
    None
}

/// End index (one past the closing `}}`) of template interpolation `{{...}}` starting at `start`.
#[inline]
fn template_var_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start + 4 > bytes.len() || bytes[start] != b'{' || bytes[start + 1] != b'{' {
        return None;
    }
    let rest = &bytes[start + 2..];
    rest.windows(2).position(|w| w == b"}}").map(|pos| start + 2 + pos + 2)
}

/// Checks if the double quote at `start` is preceded by an assignment operator `[A-Za-z0-9_]=-?`
/// (e.g. `KEY="value"`, `SECRET="`), and if so, returns the end index of the quoted literal.
#[inline]
fn code_assign_quote_end(bytes: &[u8], start: usize) -> Option<usize> {
    if start == 0 || bytes[start] != b'"' {
        return None;
    }
    let mut p = start;
    if p > 0 && bytes[p - 1] == b'=' {
        p -= 1;
        let mut id_len = 0;
        while p > 0
            && (bytes[p - 1].is_ascii_alphanumeric()
                || bytes[p - 1] == b'_'
                || bytes[p - 1] == b'-')
        {
            p -= 1;
            id_len += 1;
        }
        if id_len > 0 {
            let mut j = start + 1;
            while j < bytes.len() {
                if bytes[j] == b'\\' {
                    j += 2;
                } else if bytes[j] == b'"' {
                    return Some(j + 1);
                } else if bytes[j] == b'\n' || bytes[j] == b'\r' {
                    break;
                } else {
                    j += 1;
                }
            }
            return Some(start + 1);
        }
    }
    None
}
#[inline]
fn next_after_punct_byte(bytes: &[u8], i: usize) -> usize {
    let after = i + 1;
    if after < bytes.len() && bytes[after] == b' ' {
        after + 1
    } else {
        after
    }
}

#[inline]
pub(crate) fn get_language_rule_profile(
    lang: &str,
) -> Option<std::sync::Arc<crate::engine::rules::LanguageRuleProfile>> {
    let registry = crate::engine::rules::RuleRegistry::global();
    if let Some(prof) = registry.get(lang) {
        return Some(prof);
    }
    static INIT_ONCE: std::sync::Once = std::sync::Once::new();
    INIT_ONCE.call_once(|| {
        crate::engine::rules::RuleRegistry::auto_detect_and_init(None);
    });
    registry.get(lang)
}
/// Apply data-driven typography rules, fullwidth punctuation, and truecasing to translated output.
/// Fast-path returns Cow::Borrowed with 0 heap allocation when no transformations are required.
pub fn postprocess_typography<'a>(
    text: std::borrow::Cow<'a, str>,
    to_lang: &str,
) -> std::borrow::Cow<'a, str> {
    let mut cur = text;

    // 1. Query LanguageRuleProfile for target language
    let profile = get_language_rule_profile(to_lang);

    // 2. Chinese typography: fused fullwidth punctuation + Pangu spacing
    if to_lang.starts_with("zh") {
        let pangu_enabled = profile
            .as_ref()
            .map(|p| p.typography.pangu_spacing)
            .unwrap_or(true);
        if pangu_enabled {
            cur = apply_zh_punct_and_pangu(cur);
        } else {
            cur = apply_zh_punct_marks(cur);
        }
    } else {
        // Pre-truecasing typography transformations (Spanish inverted marks & Pangu spacing for non-zh)
        if let Some(prof) = &profile {
            if prof.typography.pangu_spacing {
                cur = apply_pangu_spacing(cur);
            }
            if prof.typography.spanish_inverted {
                cur = format_spanish_inverted(cur);
            }
        }
    }
    // 4. Western truecasing (capitalizes first letter of sentences and inverted clauses ¿/¡)
    if !to_lang.starts_with("zh") {
        cur = apply_truecase_western(cur, to_lang);
    }

    // 5. Post-truecasing typography transformations (French spacing & quotes)
    if let Some(prof) = &profile {
        if prof.typography.french_spacing {
            cur = format_french_spacing(cur);
        }
        if let Some(quote_style) = &prof.typography.quote_style {
            cur = apply_quote_style(cur, quote_style);
        }
    }
    cur
}

#[inline]
fn is_technical_or_code_punctuation(s: &str, i: usize, b: u8) -> bool {
    let bytes = s.as_bytes();
    let len = bytes.len();

    if b == b'?' {
        // Optional chaining `?.`, `?.[`, `?.(`
        if i + 1 < len && bytes[i + 1] == b'.' {
            return true;
        }
    } else if b == b'!' {
        // Non-null assertion or field access `!.`
        if i + 1 < len && bytes[i + 1] == b'.' {
            return true;
        }

        // Method or macro call with parentheses or brackets: `save!(`, `vec![`
        if i + 1 < len
            && matches!(bytes[i + 1], b'(' | b'[' | b'{')
            && i > 0
            && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')
        {
            return true;
        }

        // Method name ending with `!` (Ruby, Rails, Rust macro, Lisp)
        if i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_') {
            let mut w = i - 1;
            let mut has_underscore = bytes[w] == b'_';
            while w > 0 && (bytes[w - 1].is_ascii_alphanumeric() || bytes[w - 1] == b'_') {
                w -= 1;
                if bytes[w] == b'_' {
                    has_underscore = true;
                }
            }
            if has_underscore {
                return true;
            }
            if w > 0 && bytes[w - 1] == b'.' {
                return true;
            }
            let word = &s[w..i];
            let is_lowercase_ident = word.chars().all(|c| c.is_ascii_lowercase());
            let next_is_lowercase = s[i + 1..]
                .trim_start()
                .chars()
                .next()
                .map(|c| c.is_ascii_lowercase())
                .unwrap_or(false);
            if is_lowercase_ident && next_is_lowercase {
                return true;
            }
            if matches!(
                word,
                "save" | "create" | "update" | "destroy" | "reload" | "validate" | "find" | "delete"
                    | "reset" | "build" | "touch" | "toggle"
            ) && (i + 1 == len
                || matches!(bytes[i + 1], b',' | b';' | b':' | b'=' | b'&' | b'|' | b')' | b']' | b' ' | b'\t'))
            {
                return true;
            }
        }
    }

    false
}

/// French Typography Spacing: Intelligently inserts narrow non-breaking space `\u{202F}`
/// before French two-part punctuation marks (?, !, ;) and non-breaking space `\u{00A0}`
/// before colon (:), strictly protecting URLs, Markdown code blocks, and HTML tags.
/// Fast-path returns Cow::Borrowed when no formatting is needed (0 heap allocation).
pub fn format_french_spacing<'a>(text: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'a, str> {
    let s = text.as_ref();
    if s.is_empty() {
        return text;
    }
    if !s.bytes().any(|b| matches!(b, b'?' | b'!' | b';' | b':'))
        && !s.contains('«')
        && !s.contains('»')
    {
        return text;
    }

    let bytes = s.as_bytes();
    let mut out: Option<String> = None;
    let mut last_copied = 0;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];

        if b == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                i = tag_end;
                continue;
            }
        } else if b == b'&' {
            if let Some(ent_end) = html_entity_end(bytes, i) {
                i = ent_end;
                continue;
            }
        } else if b == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                i = code_end;
                continue;
            }
        } else if let Some(u_end) = url_end(bytes, i) {
            i = u_end;
            continue;
        }
        if !s.is_char_boundary(i) {
            i += 1;
            continue;
        }

        if s[i..].starts_with('«') {
            let ch_len = '«'.len_utf8();
            let next_pos = i + ch_len;
            let mut ws_end = next_pos;
            while ws_end < s.len() {
                let nc = s[ws_end..].chars().next().unwrap();
                if nc.is_whitespace() {
                    ws_end += nc.len_utf8();
                } else {
                    break;
                }
            }

            if ws_end > next_pos {
                if &s[next_pos..ws_end] != "\u{202F}" {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    let buf = out.as_mut().unwrap();
                    buf.push_str(&s[last_copied..next_pos]);
                    buf.push('\u{202F}');
                    last_copied = ws_end;
                }
                i = ws_end;
                continue;
            } else {
                let next_ch = s[next_pos..].chars().next();
                if next_ch.is_some() && next_ch != Some('»') {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    let buf = out.as_mut().unwrap();
                    buf.push_str(&s[last_copied..next_pos]);
                    buf.push('\u{202F}');
                    last_copied = next_pos;
                }
                i += ch_len;
                continue;
            }
        }

        if s[i..].starts_with('»') {
            let ch_len = '»'.len_utf8();
            let mut ws_start = i;
            while ws_start > 0 {
                let pc = s[..ws_start].chars().next_back().unwrap();
                if pc.is_whitespace() {
                    ws_start -= pc.len_utf8();
                } else {
                    break;
                }
            }

            if ws_start < i {
                if &s[ws_start..i] != "\u{202F}" {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    let buf = out.as_mut().unwrap();
                    buf.push_str(&s[last_copied..ws_start]);
                    buf.push('\u{202F}');
                    last_copied = i;
                }
            } else {
                let prev_ch = s[..i].chars().next_back();
                if prev_ch.is_some() && prev_ch != Some('«') {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    let buf = out.as_mut().unwrap();
                    buf.push_str(&s[last_copied..i]);
                    buf.push('\u{202F}');
                    last_copied = i;
                }
            }
            i += ch_len;
            continue;
        }

        if b == b':' {
            let colon_before = i > 0 && bytes[i - 1] == b':';
            let colon_after = i + 1 < bytes.len() && bytes[i + 1] == b':';
            let scheme_after = i + 2 < bytes.len() && bytes[i + 1] == b'/' && bytes[i + 2] == b'/';
            let alnum_before = i > 0 && bytes[i - 1].is_ascii_alphanumeric();
            let digit_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
            let hex_before = i > 0 && bytes[i - 1].is_ascii_hexdigit();
            let hex_after = i + 1 < bytes.len() && bytes[i + 1].is_ascii_hexdigit();
            let bracket_port = is_bracket_port(bytes, i);
            let line_header = is_line_head_header(bytes, i);
            if colon_before
                || colon_after
                || scheme_after
                || (alnum_before && digit_after)
                || (hex_before && hex_after)
                || bracket_port
                || line_header
            {
                i += 1;
                continue;
            }
        }

        if matches!(b, b'?' | b'!' | b';' | b':') {
            if is_technical_or_code_punctuation(s, i, b) {
                i += 1;
                continue;
            }
            // Guard: consecutive punctuation marks like `?!` or `!!`
            if matches!(b, b'?' | b'!' | b';') && i > 0 && matches!(bytes[i - 1], b'?' | b'!' | b';') {
                i += 1;
                continue;
            }

            let target_space = if b == b':' { '\u{00A0}' } else { '\u{202F}' };

            if i == 0 {
                i += 1;
                continue;
            }

            // Check whitespace preceding `i`
            let mut ws_start = i;
            while ws_start > 0 {
                let prev_ch = s[..ws_start].chars().next_back().unwrap();
                if prev_ch.is_whitespace() {
                    ws_start -= prev_ch.len_utf8();
                } else {
                    break;
                }
            }

            if ws_start < i {
                // Whitespace is present
                let ws = &s[ws_start..i];
                let mut ws_chars = ws.chars();
                let is_already_target = ws_chars.next() == Some(target_space) && ws_chars.next().is_none();
                if !is_already_target {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    let buf = out.as_mut().unwrap();
                    buf.push_str(&s[last_copied..ws_start]);
                    buf.push(target_space);
                    last_copied = i;
                }
            } else {
                // No whitespace preceding `i`
                let prev_ch = s[..i].chars().next_back().unwrap();
                if !matches!(prev_ch, '(' | '[' | '{' | '«' | '‹' | '"' | '\'') {
                    if out.is_none() {
                        out = Some(String::with_capacity(s.len() + 16));
                    }
                    let buf = out.as_mut().unwrap();
                    buf.push_str(&s[last_copied..i]);
                    buf.push(target_space);
                    last_copied = i;
                }
            }
        }

        i += 1;
    }

    match out {
        None => text,
        Some(mut buf) => {
            if last_copied < s.len() {
                buf.push_str(&s[last_copied..]);
            }
            std::borrow::Cow::Owned(buf)
        }
    }
}

/// Spanish Inverted Punctuation: Ensures interrogative sentences ending with `?` have an
/// opening `¿` and exclamatory sentences ending with `!` have an opening `¡` at the start
/// of the clause or sentence if not already present.
/// Fast-path returns Cow::Borrowed when no punctuation insertion is needed (0 heap allocation).
pub fn format_spanish_inverted<'a>(text: std::borrow::Cow<'a, str>) -> std::borrow::Cow<'a, str> {
    let s = text.as_ref();
    if s.is_empty() {
        return text;
    }
    if !s.bytes().any(|b| b == b'?' || b == b'!') {
        return text;
    }

    let bytes = s.as_bytes();
    let mut insertions: Vec<(usize, char)> = Vec::new();

    let mut current_sentence_start = 0;
    let mut has_inverted_question = false;
    let mut has_inverted_exclamation = false;
    let mut clause_starts: Vec<(usize, usize)> = Vec::new(); // (sep_pos, content_start_after_sep)

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];

        if b == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                i = tag_end;
                continue;
            }
        } else if b == b'&' {
            if let Some(ent_end) = html_entity_end(bytes, i) {
                i = ent_end;
                continue;
            }
        } else if b == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                i = code_end;
                continue;
            }
        } else if let Some(u_end) = url_end(bytes, i) {
            i = u_end;
            continue;
        }

        if !s.is_char_boundary(i) {
            i += 1;
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        let ch_len = ch.len_utf8();

        if ch == '¿' {
            has_inverted_question = true;
        } else if ch == '¡' {
            has_inverted_exclamation = true;
        } else if matches!(ch, ',' | ':' | ';' | '—' | '–') {
            let next_content = find_non_whitespace(s, i + ch_len);
            clause_starts.push((i, next_content));
        } else if ch == '?' || ch == '!' {
            // Guard: consecutive marks like `?!` or `!!`
            let is_consecutive = i > 0 && matches!(bytes[i - 1], b'?' | b'!');
            if !is_consecutive {
                let is_question = ch == '?';
                let already_has = if is_question {
                    has_inverted_question
                } else {
                    has_inverted_exclamation
                };

                if !already_has {
                    let insert_pos = clause_starts
                        .last()
                        .map(|&(_, start)| start)
                        .filter(|&start| start < i);

                    let target_pos = insert_pos.unwrap_or_else(|| {
                        find_sentence_content_start(s, current_sentence_start, i)
                    });
                    if target_pos <= i {
                        insertions.push((target_pos, if is_question { '¿' } else { '¡' }));
                    }
                }
            }

            // Terminal mark resets sentence state
            current_sentence_start = i + ch_len;
            has_inverted_question = false;
            has_inverted_exclamation = false;
            clause_starts.clear();
        } else if ch == '.' {
            let is_decimal = i > 0 && i + 1 < bytes.len() && bytes[i - 1].is_ascii_digit() && bytes[i + 1].is_ascii_digit();
            if !is_decimal {
                current_sentence_start = i + ch_len;
                has_inverted_question = false;
                has_inverted_exclamation = false;
                clause_starts.clear();
            }
        } else if ch == '\n' || ch == '\r' {
            current_sentence_start = i + ch_len;
            has_inverted_question = false;
            has_inverted_exclamation = false;
            clause_starts.clear();
        }

        i += ch_len;
    }

    if insertions.is_empty() {
        return text;
    }

    insertions.sort_by_key(|&(pos, _)| pos);
    let mut out = String::with_capacity(s.len() + insertions.len() * 4);
    let mut last = 0;
    for (pos, ch) in insertions {
        if pos >= last {
            out.push_str(&s[last..pos]);
            out.push(ch);
            last = pos;
        }
    }
    if last < s.len() {
        out.push_str(&s[last..]);
    }
    std::borrow::Cow::Owned(out)
}

#[inline]
fn find_non_whitespace(s: &str, start: usize) -> usize {
    let mut idx = start;
    while idx < s.len() {
        if let Some(c) = s[idx..].chars().next() {
            if !c.is_whitespace() {
                break;
            }
            idx += c.len_utf8();
        } else {
            break;
        }
    }
    idx.min(s.len())
}

#[inline]
fn find_sentence_content_start(s: &str, sentence_start: usize, max_pos: usize) -> usize {
    let mut idx = sentence_start;
    let bytes = s.as_bytes();
    while idx < max_pos && idx < s.len() {
        if bytes[idx] == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, idx) {
                idx = tag_end;
                continue;
            }
        }
        if let Some(c) = s[idx..].chars().next() {
            if c.is_whitespace() || matches!(c, '"' | '\'' | '“' | '«' | '‹' | '(' | '[' | '{' | '*' | '_') {
                idx += c.len_utf8();
                continue;
            }
        }
        break;
    }
    idx.min(max_pos)
}


#[inline]
fn is_opening_punctuation(c: char) -> bool {
    matches!(
        c,
        '(' | '[' | '{' | '<' | ':' | ';' | ',' | '—' | '–' | '-' | '¿' | '¡'
            | '«' | '„' | '“' | '「' | '『' | '【' | '《' | '〈'
    )
}

#[inline]
fn is_closing_punctuation(c: char) -> bool {
    matches!(
        c,
        ')' | ']' | '}' | '>' | '.' | ',' | '!' | '?' | ':' | ';' | '—' | '–' | '-'
            | '»' | '”' | '」' | '』' | '】' | '》' | '〉'
    )
}

/// Publishing-grade paired quotation mark conversion according to target language quote style:
/// - `"curly"` -> `“` and `”` (English, Chinese, etc.)
/// - `"guillemets"` -> `«` and `»` (French, Russian)
/// - `"german"` -> `„` and `“` (German)
/// - `"corner"` -> `「` and `」` (Japanese)
///
/// Strictly protects Markdown inline/block code, HTML tags/entities, and URLs.
/// Fast-path returns Cow::Borrowed with 0 heap allocation when no convertible quotes are present.
pub fn apply_quote_style<'a>(
    text: std::borrow::Cow<'a, str>,
    quote_style: &str,
) -> std::borrow::Cow<'a, str> {
    let (open_q, close_q) = match quote_style {
        "curly" => ('“', '”'),
        "guillemets" => ('«', '»'),
        "german" => ('„', '“'),
        "corner" => ('「', '」'),
        _ => return text,
    };

    let s = text.as_ref();
    let has_straight_quote = s.bytes().any(|b| b == b'"');
    let has_curly_open = s.contains('“');
    let has_curly_close = s.contains('”');

    if !has_straight_quote && (quote_style == "curly" || (!has_curly_open && !has_curly_close)) {
        return text;
    }

    let bytes = s.as_bytes();
    let mut out: Option<String> = None;
    let mut last = 0;
    let mut i = 0;
    let mut in_quote = false;

    while i < bytes.len() {
        let b = bytes[i];

        if b == b'<' {
            if let Some(tag_end) = html_tag_end(bytes, i) {
                i = tag_end;
                continue;
            }
        } else if b == b'&' {
            if let Some(ent_end) = html_entity_end(bytes, i) {
                i = ent_end;
                continue;
            }
        } else if b == b'`' {
            if let Some(code_end) = markdown_code_end(bytes, i) {
                i = code_end;
                continue;
            }
        } else if b == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(v_end) = shell_var_end(bytes, i) {
                i = v_end;
                continue;
            }
        } else if b == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            if let Some(t_end) = template_var_end(bytes, i) {
                i = t_end;
                continue;
            }
        } else if let Some(q_end) = code_assign_quote_end(bytes, i) {
            i = q_end;
            continue;
        } else if b == b'h' || b == b'f' {
            if let Some(u_end) = url_end(bytes, i) {
                i = u_end;
                continue;
            }
        }

        if b == b'\n' || b == b'\r' {
            in_quote = false;
            i += 1;
            continue;
        }

        let replacement = if b == b'"' {
            let is_inch_or_second = (i > 0 && bytes[i - 1].is_ascii_digit())
                || (i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit());
            if is_inch_or_second {
                None
            } else {
                let prev_char = s[..i].chars().next_back();
                let next_char = s[i + 1..].chars().next();

                let is_open_ctx = prev_char.map_or(true, |c| c.is_whitespace() || is_opening_punctuation(c))
                    && next_char.is_some_and(|c| !c.is_whitespace());
                let is_close_ctx = next_char.map_or(true, |c| c.is_whitespace() || is_closing_punctuation(c))
                    && prev_char.is_some_and(|c| !c.is_whitespace());

                let ch = if is_open_ctx && !is_close_ctx {
                    in_quote = true;
                    open_q
                } else if is_close_ctx && !is_open_ctx {
                    in_quote = false;
                    close_q
                } else if !in_quote {
                    in_quote = true;
                    open_q
                } else {
                    in_quote = false;
                    close_q
                };
                Some((ch, i + 1))
            }
        } else if s[i..].starts_with('“') {
            let ch_len = '“'.len_utf8();
            if quote_style == "german" {
                if in_quote {
                    in_quote = false;
                    None
                } else {
                    in_quote = true;
                    Some((open_q, i + ch_len))
                }
            } else if quote_style == "curly" {
                in_quote = true;
                None
            } else {
                in_quote = true;
                Some((open_q, i + ch_len))
            }
        } else if s[i..].starts_with('”') {
            let ch_len = '”'.len_utf8();
            if quote_style == "curly" {
                in_quote = false;
                None
            } else {
                in_quote = false;
                Some((close_q, i + ch_len))
            }
        } else if (quote_style == "german" && s[i..].starts_with('„'))
            || (quote_style == "guillemets" && s[i..].starts_with('«'))
            || (quote_style == "corner" && s[i..].starts_with('「'))
        {
            in_quote = true;
            None
        } else if (quote_style == "guillemets" && s[i..].starts_with('»'))
            || (quote_style == "corner" && s[i..].starts_with('」'))
        {
            in_quote = false;
            None
        } else {
            None
        };

        if let Some((rep_ch, next_i)) = replacement {
            let buf = out.get_or_insert_with(|| {
                let mut res = String::with_capacity(s.len() + 16);
                res.push_str(&s[..last]);
                res
            });
            if i > last {
                buf.push_str(&s[last..i]);
            }
            buf.push(rep_ch);
            i = next_i;
            last = next_i;
        } else {
            let ch_len = s[i..].chars().next().map_or(1, |c| c.len_utf8());
            i += ch_len;
        }
    }

    match out {
        None => text,
        Some(mut buf) => {
            if last < s.len() {
                buf.push_str(&s[last..]);
            }
            std::borrow::Cow::Owned(buf)
        }
    }
}

/// Make room for one more engine under `max_loaded_models`.
///
/// Called with the pool write lock held just before a new engine is
/// inserted. While the pool is at (or over) the limit, unloads the
/// least-recently-used model that may not stay resident (see
/// [`should_keep_resident`]), aborting its idle task. Protected models are
/// skipped; if every loaded model is protected the pool temporarily exceeds
/// the limit (the periodic residency cycle will still unload idlers).
fn enforce_capacity(map: &mut HashMap<String, EngineInfo>) {
    let limit = config::get().max_loaded_models;
    if limit == 0 || map.len() < limit {
        return;
    }

    // Least-recently-used first: `last_used_ms` is atomic, so reading it
    // here under the write lock gives a consistent snapshot. Keys are
    // cloned so the victim can be removed from the map below.
    let mut candidates: Vec<(String, u64)> = map
        .iter()
        .map(|(k, info)| {
            (
                k.clone(),
                info.last_used_ms(),
            )
        })
        .collect();
    candidates.sort_unstable_by_key(|&(_, used)| used);

    for (victim, last_used) in candidates {
        if should_keep_resident(&victim) {
            continue;
        }
        // Remove the victim and abort its idle task. The task's next wake
        // finds the key gone and exits by itself; abort just reclaims the
        // task slot immediately.
        if let Some(info) = map.remove(&victim) {
            if let Some(task) = info.idle_task {
                task.abort();
            }
            tracing::info!(
                "Unloaded model {} to free a slot (max-loaded-models={}); idle {} ms",
                victim,
                limit,
                EngineInfo::now_ms().saturating_sub(last_used)
            );
        }
        // One free slot is enough: the caller inserts exactly one engine.
        return;
    }
}

/// Evict a single engine from the pool by language pair key ("from-to").
///
/// Aborts its idle timer, removes the engine from the pool, and waits for any
/// in-flight translations to finish (by tracking the `Arc` strong count) so that
/// memory-mapped file handles are released before on-disk directory replacement.
pub async fn evict_engine(from_lang: &str, to_lang: &str) {
    let engine = with_key(from_lang, to_lang, |k| {
        let mut guard = ENGINES.write();
        guard.as_mut().and_then(|map| map.remove(k)).map(|info| {
            if let Some(task) = info.idle_task {
                task.abort();
            }
            info.engine
        })
    });
    if let Some(engine) = engine {
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(5);
        while Arc::strong_count(&engine) > 1 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        }
        drop(engine);
    }
}

/// Translate through a directly-held engine, bypassing the result cache.
///
/// Used by the built-in `--bench-qps` benchmark: identical engine path as
/// `translate_single_language_text` (gate + blocking pool) but every call
/// performs the full forward pass instead of hitting cached results.
pub async fn translate_uncached(
    from_lang: &str,
    to_lang: &str,
    text: &str,
) -> Result<String, ServiceError> {
    let (engine, _) = ensure_engine_loaded(from_lang, to_lang).await?;
    engine.record_call();
    let _permit = engine_gate()
        .acquire()
        .await
        .map_err(|e| ServiceError::Internal(format!("Engine gate closed: {}", e)))?;
    let text = text.to_string();
    tokio::task::spawn_blocking(move || engine.translate(&text))
        .await
        .map_err(|e| ServiceError::Internal(format!("Translation worker panicked: {}", e)))
}

/// Clean up all engines (on shutdown).
pub fn cleanup_all_engines() {
    let idle_tasks: Vec<tokio::task::JoinHandle<()>> = {
        let mut guard = ENGINES.write();
        match guard.as_mut() {
            Some(map) => map.drain().filter_map(|(_, info)| info.idle_task).collect(),
            None => Vec::new(),
        }
    };
    for task in idle_tasks {
        task.abort();
    }

    let mut guard = LOADING.lock();
    if let Some(map) = guard.as_mut() {
        map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_translated_text_deref_and_cached() {
        let arc: Arc<str> = "cached hello, world".into();
        let tt = TranslatedText::Cached(arc);
        assert_eq!(&*tt, "cached hello, world");
        assert_eq!(tt.as_str(), "cached hello, world");
        assert_eq!(format!("{}", tt), "cached hello, world");

        // Normalization on non-zh target applies truecasing
        let norm_en = tt.clone().normalize_punctuation("en");
        assert_eq!(norm_en, "Cached hello, world");

        // Normalization on zh target replaces punctuation
        let norm_zh = tt.normalize_punctuation("zh");
        assert_eq!(norm_zh, "cached hello，world");
    }

    #[tokio::test]
    async fn test_empty_text_short_circuit() {
        let auto_res = translate_with_pivot_timed_formats("auto", "zh", "", None)
            .await
            .expect("empty auto text should succeed");
        assert_eq!(auto_res.text.as_str(), "");
        assert_eq!(auto_res.detected_language, "");
        assert_eq!(auto_res.translation_duration, Duration::ZERO);
        assert_eq!(auto_res.model_load_duration, Duration::ZERO);

        let explicit_res = translate_with_pivot_timed_formats("en", "zh", "", None)
            .await
            .expect("empty explicit text should succeed");
        assert_eq!(explicit_res.text.as_str(), "");
        assert_eq!(explicit_res.detected_language, "en");
        assert_eq!(explicit_res.translation_duration, Duration::ZERO);
        assert_eq!(explicit_res.model_load_duration, Duration::ZERO);

        let seg_res = translate_segment("en", "zh", "", config::ParseFormats::default())
            .await
            .expect("empty segment should succeed");
        assert_eq!(seg_res.0, "");
        assert_eq!(seg_res.1, Duration::ZERO);
    }

    #[test]
    fn test_preload_language_pair_split() {
        assert_eq!(crate::utils::lang::split_language_pair("zh-Hans-en"), Some(("zh-Hans", "en")));
        assert_eq!(crate::utils::lang::split_language_pair("en-zh-Hans"), Some(("en", "zh-Hans")));
        assert_eq!(crate::utils::lang::split_language_pair("zh-Hans_en"), Some(("zh-Hans", "en")));
        assert_eq!(crate::utils::lang::split_language_pair("en-es"), Some(("en", "es")));
        assert_eq!(crate::utils::lang::split_language_pair("invalid"), None);
    }

    #[tokio::test]
    async fn test_loading_cleanup_guard_atomic_promise_check() {
        let test_key = "test_lang_from-test_lang_to";
        let promise1: LoadingPromise = Arc::new(tokio::sync::Mutex::new(None));
        let promise2: LoadingPromise = Arc::new(tokio::sync::Mutex::new(None));

        {
            let mut guard = LOADING.lock();
            let map = guard.get_or_insert_with(HashMap::new);
            map.insert(test_key.to_string(), promise2.clone());
        }

        // Dropping guard for promise1 should NOT remove promise2
        {
            let _guard = LoadingCleanupGuard {
                key: test_key,
                promise: &promise1,
            };
        }

        {
            let guard = LOADING.lock();
            let map = guard.as_ref().unwrap();
            assert!(map.contains_key(test_key), "promise2 must remain in LOADING");
        }

        // Dropping guard for promise2 SHOULD remove promise2
        {
            let _guard = LoadingCleanupGuard {
                key: test_key,
                promise: &promise2,
            };
        }

        {
            let guard = LOADING.lock();
            let map = guard.as_ref().unwrap();
            assert!(!map.contains_key(test_key), "promise2 must be removed after guard drop");
        }
    }

    #[tokio::test]
    async fn test_evict_engine_nonexistent() {
        if ENGINES.read().is_none() {
            super::init();
        }
        super::evict_engine("nonexistent_from", "nonexistent_to").await;
    }
}
