//! Model records: fetch from Mozilla Firefox Translations service, cache locally.
//!
//! Records are fetched from `firefox.settings.services.mozilla.com` and describe
//! available models, their versions, and download URLs.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use crate::config;
use crate::models::download::Downloader;
use crate::utils::lang::normalize_language_code;

/// A single model record from Mozilla's Firefox Translations service.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Attachment {
    pub hash: String,
    pub size: i64,
    pub filename: String,
    pub location: String,
    pub mimetype: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct RecordItem {
    /// Present in the records JSON; kept for deserialization fidelity.
    #[allow(dead_code)]
    pub name: String,
    pub schema: i64,
    pub version: String,
    #[serde(rename = "fileType")]
    pub file_type: String,
    pub attachment: Attachment,
    pub architecture: Option<String>,
    #[allow(non_snake_case)]
    pub sourceLanguage: String,
    #[allow(non_snake_case)]
    pub targetLanguage: String,
    #[serde(rename = "decompressedHash")]
    pub decompressed_hash: Option<String>,
    #[serde(rename = "decompressedSize")]
    pub decompressed_size: Option<i64>,
    pub id: String,
    #[serde(rename = "last_modified")]
    pub last_modified: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub struct LanguagePair {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub struct LanguageListResponse {
    pub languages: Vec<String>,
    pub pairs: Vec<LanguagePair>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct RecordsData {
    pub data: Vec<RecordItem>,
    #[serde(skip)]
    pub pairs_index: HashMap<String, HashSet<String>>,
    #[serde(skip)]
    pub supported_languages: HashSet<String>,
    #[serde(skip)]
    pub supported_languages_list: Vec<String>,
    #[serde(skip)]
    pub pairs_list: Vec<String>,
    #[serde(skip)]
    pub language_list_response: std::sync::Arc<LanguageListResponse>,
}

impl RecordsData {
    pub fn build_indices(&mut self) {
        let mut pairs_index: HashMap<String, HashSet<String>> = HashMap::new();
        let mut supported_languages: HashSet<String> = HashSet::new();

        for rec in &self.data {
            let from = normalize_language_code(&rec.sourceLanguage);
            let to = normalize_language_code(&rec.targetLanguage);
            supported_languages.insert(from.clone());
            supported_languages.insert(to.clone());
            pairs_index.entry(from).or_default().insert(to);
        }

        let mut supported_languages_list: Vec<String> = supported_languages.iter().cloned().collect();
        supported_languages_list.sort();

        let mut pairs_list = Vec::with_capacity(pairs_index.len() * 2);
        let mut pairs_split = Vec::with_capacity(pairs_index.len() * 2);
        for (from, targets) in &pairs_index {
            for to in targets {
                pairs_list.push(format!("{}_{}", from, to));
                pairs_split.push(LanguagePair {
                    from: from.clone(),
                    to: to.clone(),
                });
            }
        }
        pairs_list.sort();
        pairs_split.sort_by(|a, b| (&a.from, &a.to).cmp(&(&b.from, &b.to)));

        let response = LanguageListResponse {
            languages: supported_languages_list.clone(),
            pairs: pairs_split,
        };

        self.pairs_index = pairs_index;
        self.supported_languages = supported_languages;
        self.supported_languages_list = supported_languages_list;
        self.pairs_list = pairs_list;
        self.language_list_response = std::sync::Arc::new(response);
    }
}
/// Installed model records, swapped atomically so the background updater can
/// refresh the table while request threads take a consistent snapshot.
static GLOBAL_RECORDS: parking_lot::RwLock<Option<std::sync::Arc<RecordsData>>> =
    parking_lot::RwLock::new(None);

const RECORDS_URL: &str = "https://firefox.settings.services.mozilla.com/v1/buckets/main-preview/collections/translations-models-v2/records";
const ATTACHMENTS_BASE_URL: &str = "https://firefox-settings-attachments.cdn.mozilla.net";

/// Initialize model records (load from local cache if available, or fetch from remote if missing).
///
/// If `records.json` exists locally, it is loaded immediately without waiting
/// on network requests. If `records.json` is missing (e.g. on a fresh install) and
/// offline mode is not enabled, records are fetched from remote to bootstrap the registry.
/// When `--check-model-update` is enabled, the background updater refreshes
/// `records.json` and checks for model updates asynchronously.
pub async fn init_records() -> Result<(), String> {
    let cfg = config::get();

    // Always ensure the config and model directories exist before writing/reading
    // records and models (mirrors `fs.mkdir(..., { recursive: true })` in the TS server).
    tokio::fs::create_dir_all(&cfg.config_dir)
        .await
        .map_err(|e| format!("Failed to create config dir: {}", e))?;
    tokio::fs::create_dir_all(&cfg.model_dir)
        .await
        .map_err(|e| format!("Failed to create model dir: {}", e))?;
    cleanup_orphan_staging(&cfg.model_dir).await;

    let records_path = cfg.config_dir.join("records.json");
    if records_path.exists() {
        match load_records_from_file(&cfg.config_dir).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                if cfg.enable_offline_mode {
                    return Err(e);
                }
                tracing::warn!(
                    "Failed to parse local records file: {}, fetching from remote...",
                    e
                );
            }
        }
    } else if cfg.enable_offline_mode {
        return Err(format!(
            "Offline mode enabled and model records file not found: {}",
            records_path.display()
        ));
    }

    fetch_records(&cfg.config_dir).await
}

/// Download the latest records and update the global table.
///
/// Also persists the raw JSON to `records.json` for offline use. Network or
/// parse errors are surfaced to the caller; the in-memory table is only
/// replaced once a fully valid payload is available.
pub async fn fetch_records(config_dir: &Path) -> Result<(), String> {
    let downloader = Downloader::new();
    let response = downloader
        .client()
        .get(RECORDS_URL)
        .header(
            reqwest::header::USER_AGENT,
            concat!("MTranServer/", env!("CARGO_PKG_VERSION")),
        )
        .send()
        .await
        .map_err(|e| format!("Failed to download records: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "HTTP {}: Failed to fetch records from Mozilla",
            response.status().as_u16()
        ));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read records response: {}", e))?;

    // Parse to validate before writing
    let records: RecordsData =
        serde_json::from_slice(&bytes).map_err(|e| format!("Failed to parse records: {}", e))?;

    // Write to file for offline use (atomic: temp file + rename).
    let records_path = config_dir.join("records.json");
    crate::models::download::write_atomic(&records_path, &bytes).await?;

    set_records(records);

    Ok(())
}

#[inline]
pub(crate) fn record_matches_pair(record: &RecordItem, from_lang: &str, to_lang: &str) -> bool {
    let norm_from = crate::utils::lang::normalize_language_code_cow(from_lang);
    let norm_to = crate::utils::lang::normalize_language_code_cow(to_lang);
    let rec_from = crate::utils::lang::normalize_language_code_cow(&record.sourceLanguage);
    let rec_to = crate::utils::lang::normalize_language_code_cow(&record.targetLanguage);

    (record.sourceLanguage == from_lang || rec_from == norm_from)
        && (record.targetLanguage == to_lang || rec_to == norm_to)
}

async fn load_records_from_file(config_dir: &Path) -> Result<(), String> {
    let path = config_dir.join("records.json");
    let data = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Failed to read records file: {}", e))?;
    let records: RecordsData =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse records: {}", e))?;
    set_records(records);
    Ok(())
}

fn set_records(mut records: RecordsData) {
    records.build_indices();
    *GLOBAL_RECORDS.write() = Some(std::sync::Arc::new(records));
}

fn with_records<R>(f: impl FnOnce(&RecordsData) -> R) -> Option<R> {
    GLOBAL_RECORDS.read().as_deref().map(f)
}

fn records() -> Option<std::sync::Arc<RecordsData>> {
    GLOBAL_RECORDS.read().clone()
}


/// Check if a language pair is available in the records.
pub fn has_language_pair(from_lang: &str, to_lang: &str) -> bool {
    let from = crate::utils::lang::normalize_language_code_cow(from_lang);
    let to = crate::utils::lang::normalize_language_code_cow(to_lang);
    with_records(|r| {
        r.pairs_index
            .get(from.as_ref())
            .is_some_and(|targets| targets.contains(to.as_ref()))
    })
    .unwrap_or(false)
}

/// Get all available language pairs as "from_to" strings.
pub fn get_language_pairs() -> Vec<String> {
    with_records(|r| r.pairs_list.clone()).unwrap_or_default()
}

/// Get all supported language codes.
pub fn get_supported_languages() -> Vec<String> {
    with_records(|r| r.supported_languages_list.clone()).unwrap_or_default()
}

/// Get the cached language list response for `/languages` without recomputing.
pub fn get_language_list_response() -> std::sync::Arc<LanguageListResponse> {
    static EMPTY: std::sync::LazyLock<std::sync::Arc<LanguageListResponse>> =
        std::sync::LazyLock::new(|| std::sync::Arc::new(LanguageListResponse::default()));
    with_records(|r| std::sync::Arc::clone(&r.language_list_response))
        .unwrap_or_else(|| std::sync::Arc::clone(&EMPTY))
}

/// Check if a language code has at least one model record (source or target).
pub fn is_supported_language(code: &str) -> bool {
    let code = crate::utils::lang::normalize_language_code_cow(code);
    with_records(|r| r.supported_languages.contains(code.as_ref())).unwrap_or(true)
}

struct StagingGuard {
    path: std::path::PathBuf,
    committed: bool,
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.committed && self.path.exists() {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

async fn cleanup_orphan_staging(model_dir: &Path) {
    let Ok(mut entries) = tokio::fs::read_dir(model_dir).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') && (name.contains(".staging") || name.ends_with(".bak")) {
            let path = entry.path();
            if let Ok(ft) = entry.file_type().await {
                if ft.is_dir() {
                    let _ = tokio::fs::remove_dir_all(&path).await;
                }
            }
        }
    }
}

fn pair_download_lock(pair: &str) -> Arc<tokio::sync::Mutex<()>> {
    static LOCKS: parking_lot::Mutex<Option<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
        parking_lot::Mutex::new(None);
    let mut guard = LOCKS.lock();
    let map = guard.get_or_insert_with(HashMap::new);
    map.entry(pair.to_string())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// Download a model for a specific language pair to the given model directory.
pub async fn download_model_dir(
    model_dir: &Path,
    from_lang: &str,
    to_lang: &str,
    version: Option<&str>,
) -> Result<(), String> {
    let norm_from = crate::utils::lang::normalize_language_code_cow(from_lang);
    let norm_to = crate::utils::lang::normalize_language_code_cow(to_lang);
    let pair_name = format!("{}_{}", norm_from, norm_to);
    let _lock = pair_download_lock(&pair_name).lock_owned().await;

    let lang_pair_dir = model_dir.join(&pair_name);

    if let Ok(files) = get_model_files(model_dir, from_lang, to_lang).await {
        if !files.is_empty() {
            if let Some(target_ver) = version {
                if let Ok(Some(installed)) = read_installed_version(&lang_pair_dir).await {
                    if installed.get("model").is_some_and(|v| v == target_ver) {
                        return Ok(());
                    }
                }
            } else if let Some(latest) = latest_record_versions(from_lang, to_lang) {
                if let Ok(Some(installed)) = read_installed_version(&lang_pair_dir).await {
                    let up_to_date = latest.iter().all(|(ft, ver)| {
                        installed.get(ft).is_some_and(|cur| {
                            compare_versions(cur, ver) != std::cmp::Ordering::Less
                        })
                    });
                    if up_to_date {
                        return Ok(());
                    }
                }
            }
        }
    }

    static STAGING_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let nonce = STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let staging_dir = model_dir.join(format!(".{pair_name}.staging.{}_{}", std::process::id(), nonce));

    tokio::fs::create_dir_all(&staging_dir)
        .await
        .map_err(|e| format!("Failed to create staging dir: {}", e))?;

    let mut guard = StagingGuard {
        path: staging_dir.clone(),
        committed: false,
    };

    let downloader = Downloader::new();
    let existing = if lang_pair_dir.exists() {
        Some(lang_pair_dir.as_path())
    } else {
        None
    };

    let downloaded = download_model_to_dir(
        &downloader,
        &staging_dir,
        from_lang,
        to_lang,
        version,
        existing,
    )
    .await?;

    write_installed_version(&staging_dir, &downloaded).await?;

    crate::service::evict_engine(from_lang, to_lang).await;

    crate::models::download::swap_dir_atomic(&staging_dir, &lang_pair_dir).await?;
    guard.committed = true;

    crate::service::evict_engine(from_lang, to_lang).await;
    Ok(())
}

/// Download a model for a specific language pair.
///
/// Downloads the latest version of the model files for the given language
/// pair, skipping files that are already up-to-date (matching hash).
/// Records the installed versions for future update checks.
pub async fn download_model(
    from_lang: &str,
    to_lang: &str,
    version: Option<&str>,
) -> Result<(), String> {
    let cfg = config::get();
    download_model_dir(&cfg.model_dir, from_lang, to_lang, version).await
}

/// Core download logic: fetch model files for a language pair into the given
/// directory, checking hashes to skip already-downloaded files.
///
/// Every file is downloaded/decompressed through a `.part` temp name, verified
/// against the record's `decompressedHash` (when present) and only then moved
/// into place — a corrupted download can never be registered as installed.
///
/// Returns a map of `fileType → version` for every file that was processed
/// (either newly downloaded or already up-to-date).
pub(crate) async fn download_model_to_dir(
    downloader: &Downloader,
    target_dir: &Path,
    from_lang: &str,
    to_lang: &str,
    version: Option<&str>,
    existing_dir: Option<&Path>,
) -> Result<HashMap<String, String>, String> {
    let records = records().ok_or("Records not initialized")?;

    let mut matched: Vec<&RecordItem> = records
        .data
        .iter()
        .filter(|r| {
            let rec_source = crate::utils::lang::normalize_language_code_cow(&r.sourceLanguage);
            let rec_target = crate::utils::lang::normalize_language_code_cow(&r.targetLanguage);
            let match_source = r.sourceLanguage == from_lang || rec_source.as_ref() == from_lang;
            let match_target = r.targetLanguage == to_lang || rec_target.as_ref() == to_lang;
            match_source && match_target && version.map_or(true, |v| r.version == v)
        })
        .collect();

    if matched.is_empty() {
        return Err(format!("No model found for {} -> {}", from_lang, to_lang));
    }

    // When no version is requested, pick only the latest version per fileType.
    if version.is_none() {
        let mut file_type_map: HashMap<String, Vec<&RecordItem>> = HashMap::new();
        for record in &matched {
            file_type_map
                .entry(record.file_type.clone())
                .or_default()
                .push(record);
        }

        let mut target_records: Vec<&RecordItem> = Vec::new();
        for (_, recs) in file_type_map {
            let latest = recs
                .iter()
                .max_by(|a, b| compare_versions(&a.version, &b.version))
                .expect("file_type group is non-empty");
            target_records.push(latest);
        }
        matched = target_records;
    }

    tokio::fs::create_dir_all(target_dir)
        .await
        .map_err(|e| format!("Failed to create model dir: {}", e))?;

    let mut downloaded_versions: HashMap<String, String> = HashMap::new();

    for &record in &matched {
        let filename = &record.attachment.filename;
        let file_url = format!("{}/{}", ATTACHMENTS_BASE_URL, record.attachment.location);
        let decompressed_filename = filename.trim_end_matches(".zst");
        let decompressed_path = target_dir.join(decompressed_filename);

        // Reuse unchanged file from existing_dir via hard link or copy
        let mut reused = false;
        if !decompressed_path.exists() {
            if let Some(src_dir) = existing_dir {
                let src_path = src_dir.join(decompressed_filename);
                if src_path.exists() {
                    let mut matched_hash = false;
                    let size_matches = record
                        .decompressed_size
                        .map_or(true, |sz| {
                            std::fs::metadata(&src_path).map(|m| m.len() == sz as u64).unwrap_or(false)
                        });
                    if size_matches {
                        if let (Some(expected), Ok(actual)) =
                            (&record.decompressed_hash, compute_file_hash(&src_path).await)
                        {
                            matched_hash = *expected == actual;
                        }
                    }
                    if matched_hash
                        && (tokio::fs::hard_link(&src_path, &decompressed_path).await.is_ok()
                            || tokio::fs::copy(&src_path, &decompressed_path).await.is_ok())
                    {
                        reused = true;
                    }
                }
            }
        }

        let needs_download = if reused {
            false
        } else if decompressed_path.exists() {
            match &record.decompressed_hash {
                Some(expected_hash) => {
                    compute_file_hash(&decompressed_path).await? != *expected_hash
                }
                None => false,
            }
        } else {
            true
        };

        if !needs_download {
            downloaded_versions.insert(record.file_type.clone(), record.version.clone());
            continue;
        }

        downloader
            .download_stream_decompressed(
                &file_url,
                &decompressed_path,
                filename.ends_with(".zst"),
                record.decompressed_size.map(|s| s as u64),
                record.decompressed_hash.as_deref(),
            )
            .await?;

        downloaded_versions.insert(record.file_type.clone(), record.version.clone());
    }

    Ok(downloaded_versions)
}

/// Get the model file paths for a language pair.
pub async fn get_model_files(
    model_dir: &Path,
    from_lang: &str,
    to_lang: &str,
) -> Result<HashMap<String, String>, String> {
    let records = records().ok_or("Records not initialized")?;

    let lang_pair_dir = model_dir.join(format!("{}_{}", from_lang, to_lang));
    let mut file_type_map: HashMap<String, String> = HashMap::new();

    for record in &records.data {
        if record_matches_pair(record, from_lang, to_lang) {
            let filename = record.attachment.filename.trim_end_matches(".zst");
            let full_path = lang_pair_dir.join(filename);
            if full_path.exists() {
                file_type_map.insert(
                    record.file_type.clone(),
                    full_path.to_string_lossy().into_owned(),
                );
            }
        }
    }

    let mut files = HashMap::new();

    let model_path = file_type_map
        .get("model")
        .ok_or_else(|| format!("Model file not found for {} -> {}", from_lang, to_lang))?;
    files.insert("model".to_string(), model_path.clone());

    let lex_path = file_type_map
        .get("lex")
        .ok_or_else(|| format!("Lex file not found for {} -> {}", from_lang, to_lang))?;
    files.insert("lex".to_string(), lex_path.clone());

    let has_paired_vocabs = file_type_map.contains_key("srcvocab") && file_type_map.contains_key("trgvocab");
    let installed_versions = read_installed_version(&lang_pair_dir).await.ok().flatten();

    let use_split_vocab = if let Some(ref installed) = installed_versions {
        if installed.contains_key("srcvocab") && installed.contains_key("trgvocab") && has_paired_vocabs {
            true
        } else if installed.contains_key("vocab") && file_type_map.contains_key("vocab") {
            false
        } else {
            has_paired_vocabs
        }
    } else {
        has_paired_vocabs
    };

    if use_split_vocab {
        let src_vocab = file_type_map
            .get("srcvocab")
            .ok_or_else(|| format!("Source vocab not found for {} -> {}", from_lang, to_lang))?;
        files.insert("vocab_src".to_string(), src_vocab.clone());

        let trg_vocab = file_type_map
            .get("trgvocab")
            .ok_or_else(|| format!("Target vocab not found for {} -> {}", from_lang, to_lang))?;
        files.insert("vocab_trg".to_string(), trg_vocab.clone());
    } else if let Some(vocab_path) = file_type_map.get("vocab") {
        files.insert("vocab_src".to_string(), vocab_path.clone());
        files.insert("vocab_trg".to_string(), vocab_path.clone());
    } else {
        return Err(format!(
            "Neither paired vocabs (srcvocab/trgvocab) nor unified vocab found for {} -> {}",
            from_lang, to_lang
        ));
    }

    Ok(files)
}

use super::download::compute_file_hash;

/// Compare two version strings using semantic versioning rules.
pub(crate) use crate::utils::compare_versions;

/// Version metadata persisted next to a downloaded language pair.
///
/// Maps each `fileType` ("model", "lex", "vocab", "srcvocab", "trgvocab")
/// to the installed record version.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InstalledVersion {
    pub versions: HashMap<String, String>,
    pub updated_at: i64,
}

/// Write the installed versions for a language pair directory.
pub async fn write_installed_version(
    pair_dir: &Path,
    versions: &HashMap<String, String>,
) -> Result<(), String> {
    let path = pair_dir.join("version.json");
    let data = InstalledVersion {
        versions: versions.clone(),
        updated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
    };
    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| format!("Failed to serialize version info: {}", e))?;
    crate::models::download::write_atomic(&path, json.as_bytes()).await
}

/// Read the installed versions for a language pair directory, if any.
pub async fn read_installed_version(
    pair_dir: &Path,
) -> Result<Option<HashMap<String, String>>, String> {
    let path = pair_dir.join("version.json");
    if !path.exists() {
        return Ok(None);
    }
    let data = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| format!("Failed to read version file: {}", e))?;
    let parsed: InstalledVersion =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse version file: {}", e))?;
    Ok(Some(parsed.versions))
}

/// List language pairs that have at least one model file installed on disk.
///
/// A pair is "installed" when its `<from>_<to>` directory exists and contains
/// a `model` fileType (the `.bin` file). Used by the updater to scope its
/// checks to what the user already has.
pub async fn installed_pairs(model_dir: &Path) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let Ok(mut entries) = tokio::fs::read_dir(model_dir).await else {
        return pairs;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let file_name = entry.file_name().to_string_lossy().into_owned();
        if !entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        if file_name.starts_with('.') {
            continue;
        }
        let Some(sep) = file_name.find('_') else {
            continue;
        };
        let from = &file_name[..sep];
        let to = &file_name[sep + 1..];
        if from.is_empty() || to.is_empty() {
            continue;
        }
        // Require at least the model file to consider the pair installed.
        let pair_dir = model_dir.join(&file_name);
        let has_model = {
            let mut has = false;
            if let Ok(mut files) = tokio::fs::read_dir(&pair_dir).await {
                while let Ok(Some(f)) = files.next_entry().await {
                    let name = f.file_name().to_string_lossy().into_owned();
                    if name.ends_with(".bin") {
                        has = true;
                        break;
                    }
                }
            }
            has
        };
        if has_model {
            pairs.push((from.to_string(), to.to_string()));
        }
    }
    pairs.sort();
    pairs.dedup();
    pairs
}

/// Infer installed versions for a pair by matching local file hashes against
/// the records, for pairs that predate `version.json` tracking.
///
/// Returns `None` when the pair cannot be matched (e.g. records unavailable).
pub async fn infer_installed_version(
    pair_dir: &Path,
    from_lang: &str,
    to_lang: &str,
) -> Option<HashMap<String, String>> {
    let records = records()?;
    let mut versions: HashMap<String, String> = HashMap::new();

    let mut files = tokio::fs::read_dir(pair_dir).await.ok()?;
    while let Ok(Some(entry)) = files.next_entry().await {
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().into_owned();
        if !entry
            .file_type()
            .await
            .map(|t| t.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        if !path.exists() {
            continue;
        }
        let local_hash = compute_file_hash(&path).await.ok()?;
        // Find a record for this pair whose decompressed hash matches.
        for rec in &records.data {
            if !record_matches_pair(rec, from_lang, to_lang) {
                continue;
            }
            if rec.attachment.filename.trim_end_matches(".zst") != file_name {
                continue;
            }
            if let Some(expected) = &rec.decompressed_hash {
                if *expected == local_hash {
                    versions
                        .entry(rec.file_type.clone())
                        .or_insert_with(|| rec.version.clone());
                }
            }
        }
    }

    if versions.is_empty() {
        None
    } else {
        Some(versions)
    }
}

/// Get the latest record versions for a language pair (one per fileType).
pub fn latest_record_versions(from_lang: &str, to_lang: &str) -> Option<HashMap<String, String>> {
    let records = records()?;
    let matched: Vec<&RecordItem> = records
        .data
        .iter()
        .filter(|r| record_matches_pair(r, from_lang, to_lang))
        .collect();
    if matched.is_empty() {
        return None;
    }

    let mut file_type_map: HashMap<String, Vec<&RecordItem>> = HashMap::new();
    for record in matched {
        file_type_map
            .entry(record.file_type.clone())
            .or_default()
            .push(record);
    }

    let mut latest: HashMap<String, String> = HashMap::new();
    for (file_type, records) in file_type_map {
        let newest = records
            .iter()
            .max_by(|a, b| compare_versions(&a.version, &b.version))
            .unwrap();
        latest.insert(file_type, newest.version.clone());
    }
    Some(latest)
}

#[cfg(test)]
mod tests {
    static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    #[tokio::test]
    async fn test_init_records_behavior() {
        let _lock = TEST_LOCK.lock().await;
        let temp_dir1 = std::env::temp_dir().join(format!(
            "mtran_test_records_1_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = tokio::fs::create_dir_all(&temp_dir1).await;
        let records_path = temp_dir1.join("records.json");
        tokio::fs::write(&records_path, r#"{"data":[]}"#)
            .await
            .unwrap();

        let mut cli = crate::config::Cli::parse_from(["mtranserver"]);
        cli.data_dir = temp_dir1.to_str().unwrap().to_string();
        crate::config::init(cli);

        let res = super::init_records().await;
        assert!(
            res.is_ok(),
            "Expected init_records to succeed from local file: {:?}",
            res
        );
        let _ = tokio::fs::remove_dir_all(&temp_dir1).await;

        // 2. When records.json is missing and offline mode is enabled, it should return an error without network
        let temp_dir2 = std::env::temp_dir().join(format!(
            "mtran_test_records_2_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = tokio::fs::create_dir_all(&temp_dir2).await;

        let mut cli2 = crate::config::Cli::parse_from(["mtranserver", "--offline"]);
        cli2.data_dir = temp_dir2.to_str().unwrap().to_string();
        crate::config::init(cli2);

        let res2 = super::init_records().await;
        assert!(
            res2.is_err(),
            "Expected init_records to fail when records.json is missing in offline mode: {:?}",
            res2
        );
        assert!(res2.unwrap_err().contains("Offline mode enabled"));
        let _ = tokio::fs::remove_dir_all(&temp_dir2).await;
    }

    use super::compare_versions;
    use clap::Parser;
    use std::cmp::Ordering;

    #[test]
    fn equal_versions_compare_equal() {
        assert_eq!(compare_versions("1.2.3", "1.2.3"), Ordering::Equal);
    }

    #[test]
    fn higher_component_orders_greater() {
        assert_eq!(compare_versions("1.2.4", "1.2.3"), Ordering::Greater);
        assert_eq!(compare_versions("2.0.0", "1.9.9"), Ordering::Greater);
        assert_eq!(compare_versions("1.3.0", "1.2.9"), Ordering::Greater);
    }

    #[test]
    fn missing_components_count_as_zero() {
        assert_eq!(compare_versions("1.2", "1.2.0"), Ordering::Equal);
        assert_eq!(compare_versions("1.2", "1.2.1"), Ordering::Less);
        assert_eq!(compare_versions("1.3", "1.2.9"), Ordering::Greater);
    }

    #[test]
    fn non_numeric_segments_are_dropped() {
        // A segment that fails to parse is dropped, shifting the remaining
        // parts left: "1.2beta" compares as "1" against "1.2".
        assert_eq!(compare_versions("1.2beta", "1.2"), Ordering::Less);
        assert_eq!(compare_versions("abc", "0"), Ordering::Equal);
    }

    #[test]
    fn different_lengths_pad_with_zero() {
        assert_eq!(compare_versions("1.0.0.0", "1"), Ordering::Equal);
    }
    #[test]
    fn test_has_language_pair_indexed() {
        let _lock = TEST_LOCK.blocking_lock();
        let data = super::RecordsData {
            data: vec![
                super::RecordItem {
                    name: String::new(),
                    schema: 0,
                    version: "1.0".to_string(),
                    file_type: "model".to_string(),
                    attachment: super::Attachment {
                        hash: String::new(),
                        size: 0,
                        filename: String::new(),
                        location: String::new(),
                        mimetype: String::new(),
                    },
                    architecture: None,
                    sourceLanguage: "en".to_string(),
                    targetLanguage: "es".to_string(),
                    decompressed_hash: None,
                    decompressed_size: None,
                    id: "1".to_string(),
                    last_modified: 0,
                },
                super::RecordItem {
                    name: String::new(),
                    schema: 0,
                    version: "1.0".to_string(),
                    file_type: "model".to_string(),
                    attachment: super::Attachment {
                        hash: String::new(),
                        size: 0,
                        filename: String::new(),
                        location: String::new(),
                        mimetype: String::new(),
                    },
                    architecture: None,
                    sourceLanguage: "zh".to_string(),
                    targetLanguage: "en".to_string(),
                    decompressed_hash: None,
                    decompressed_size: None,
                    id: "2".to_string(),
                    last_modified: 0,
                },
            ],
            pairs_index: std::collections::HashMap::new(),
            supported_languages: std::collections::HashSet::new(),
            ..Default::default()
        };
        super::set_records(data);
        assert!(super::has_language_pair("en", "es"));
        assert!(super::has_language_pair("EN", "ES"));
        assert!(super::has_language_pair("zh", "en"));
        assert!(super::has_language_pair("zh-CN", "en"));
        assert!(super::has_language_pair("zh-Hans", "en"));
        assert!(!super::has_language_pair("es", "en"));
        assert!(!super::has_language_pair("fr", "de"));
        assert!(super::is_supported_language("en"));
        assert!(super::is_supported_language("zh-cn"));
        assert!(!super::is_supported_language("nonexistent"));
        let pairs = super::get_language_pairs();
        assert!(pairs.contains(&"zh-Hans_en".to_string()));
        assert!(!pairs.contains(&"zh_en".to_string()));
        let langs = super::get_supported_languages();
        assert!(langs.contains(&"zh-Hans".to_string()));
        assert!(!langs.contains(&"zh".to_string()));
    }

    #[tokio::test]
    async fn test_installed_pairs_ignores_staging_and_bak() {
        let temp_dir = std::env::temp_dir().join(format!(
            "mtran_test_installed_pairs_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let valid_pair = temp_dir.join("en_es");
        let staging_pair = temp_dir.join(".en_es.staging");
        let bak_pair = temp_dir.join(".en_es.bak");
        let invalid = temp_dir.join("invalid");

        tokio::fs::create_dir_all(&valid_pair).await.unwrap();
        tokio::fs::write(valid_pair.join("model.bin"), b"model").await.unwrap();

        tokio::fs::create_dir_all(&staging_pair).await.unwrap();
        tokio::fs::write(staging_pair.join("model.bin"), b"staging model").await.unwrap();

        tokio::fs::create_dir_all(&bak_pair).await.unwrap();
        tokio::fs::write(bak_pair.join("model.bin"), b"bak model").await.unwrap();

        tokio::fs::create_dir_all(&invalid).await.unwrap();

        let pairs = super::installed_pairs(&temp_dir).await;
        assert_eq!(pairs, vec![("en".to_string(), "es".to_string())]);

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }
    #[tokio::test]
    async fn test_cleanup_orphan_staging() {
        let temp_dir = std::env::temp_dir().join(format!(
            "mtran_test_cleanup_staging_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let valid_dir = temp_dir.join("en_es");
        let orphan_staging1 = temp_dir.join(".en_es.staging");
        let orphan_staging2 = temp_dir.join(".en_es.staging.999_1");
        let orphan_bak = temp_dir.join(".en_es.bak");

        tokio::fs::create_dir_all(&valid_dir).await.unwrap();
        tokio::fs::create_dir_all(&orphan_staging1).await.unwrap();
        tokio::fs::create_dir_all(&orphan_staging2).await.unwrap();
        tokio::fs::create_dir_all(&orphan_bak).await.unwrap();

        super::cleanup_orphan_staging(&temp_dir).await;

        assert!(valid_dir.exists());
        assert!(!orphan_staging1.exists());
        assert!(!orphan_staging2.exists());
        assert!(!orphan_bak.exists());

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[test]
    fn test_staging_guard_cleanup() {
        let temp_dir = std::env::temp_dir().join(format!(
            "mtran_test_staging_guard_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let staging1 = temp_dir.join(".test.staging.1");
        let staging2 = temp_dir.join(".test.staging.2");

        std::fs::create_dir_all(&staging1).unwrap();
        std::fs::create_dir_all(&staging2).unwrap();

        {
            let _guard = super::StagingGuard {
                path: staging1.clone(),
                committed: false,
            };
        }
        assert!(!staging1.exists());

        {
            let mut guard = super::StagingGuard {
                path: staging2.clone(),
                committed: false,
            };
            guard.committed = true;
        }
        assert!(staging2.exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_concurrent_download_model_dir_coalescing() {
        let _lock = TEST_LOCK.lock().await;
        let temp_dir = std::env::temp_dir().join(format!(
            "mtran_test_concurrent_dl_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pair_dir = temp_dir.join("en_es");
        tokio::fs::create_dir_all(&pair_dir).await.unwrap();
        tokio::fs::write(pair_dir.join("model.bin"), b"dummy model").await.unwrap();
        tokio::fs::write(pair_dir.join("lex.s2t.bin"), b"dummy lex").await.unwrap();
        tokio::fs::write(pair_dir.join("vocab.spm"), b"dummy vocab").await.unwrap();

        let mut versions = std::collections::HashMap::new();
        versions.insert("model".to_string(), "1.0".to_string());
        versions.insert("lex".to_string(), "1.0".to_string());
        versions.insert("vocab".to_string(), "1.0".to_string());
        super::write_installed_version(&pair_dir, &versions).await.unwrap();

        let data = super::RecordsData {
            data: vec![
                super::RecordItem {
                    name: String::new(),
                    schema: 0,
                    version: "1.0".to_string(),
                    file_type: "model".to_string(),
                    attachment: super::Attachment {
                        hash: String::new(),
                        size: 0,
                        filename: "model.bin".to_string(),
                        location: String::new(),
                        mimetype: String::new(),
                    },
                    architecture: None,
                    sourceLanguage: "en".to_string(),
                    targetLanguage: "es".to_string(),
                    decompressed_hash: None,
                    decompressed_size: None,
                    id: "1".to_string(),
                    last_modified: 0,
                },
                super::RecordItem {
                    name: String::new(),
                    schema: 0,
                    version: "1.0".to_string(),
                    file_type: "lex".to_string(),
                    attachment: super::Attachment {
                        hash: String::new(),
                        size: 0,
                        filename: "lex.s2t.bin".to_string(),
                        location: String::new(),
                        mimetype: String::new(),
                    },
                    architecture: None,
                    sourceLanguage: "en".to_string(),
                    targetLanguage: "es".to_string(),
                    decompressed_hash: None,
                    decompressed_size: None,
                    id: "2".to_string(),
                    last_modified: 0,
                },
                super::RecordItem {
                    name: String::new(),
                    schema: 0,
                    version: "1.0".to_string(),
                    file_type: "vocab".to_string(),
                    attachment: super::Attachment {
                        hash: String::new(),
                        size: 0,
                        filename: "vocab.spm".to_string(),
                        location: String::new(),
                        mimetype: String::new(),
                    },
                    architecture: None,
                    sourceLanguage: "en".to_string(),
                    targetLanguage: "es".to_string(),
                    decompressed_hash: None,
                    decompressed_size: None,
                    id: "3".to_string(),
                    last_modified: 0,
                },
            ],
            ..Default::default()
        };
        super::set_records(data);

        let mut handles = Vec::new();
        for _ in 0..10 {
            let dir = temp_dir.clone();
            handles.push(tokio::spawn(async move {
                super::download_model_dir(&dir, "en", "es", None).await
            }));
        }

        for h in handles {
            let res = h.await.unwrap();
            assert!(res.is_ok(), "Concurrent download failed: {:?}", res);
        }

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[test]
    fn test_global_records_no_poison() {
        let _lock = TEST_LOCK.blocking_lock();
        let _ = std::panic::catch_unwind(|| {
            super::with_records(|_| {
                panic!("intentional panic while reading records");
            });
        });

        let data = super::RecordsData {
            data: vec![],
            ..Default::default()
        };
        super::set_records(data);
        assert!(super::records().is_some());
    }

    #[tokio::test]
    async fn test_reuse_existing_file_without_second_hash() {
        let _lock = TEST_LOCK.lock().await;
        let temp_dir = std::env::temp_dir().join(format!(
            "mtran_test_reuse_{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let src_dir = temp_dir.join("src");
        let target_dir = temp_dir.join("target");
        let _ = tokio::fs::create_dir_all(&src_dir).await;
        let _ = tokio::fs::create_dir_all(&target_dir).await;

        let content = b"pre-existing model data for testing reuse";
        let src_file = src_dir.join("model.bin");
        tokio::fs::write(&src_file, content).await.unwrap();

        let hash = super::compute_file_hash(&src_file).await.unwrap();

        let item = super::RecordItem {
            schema: 0,
            name: "test-model".to_string(),
            version: "1.0".to_string(),
            file_type: "model".to_string(),
            attachment: super::Attachment {
                hash: hash.clone(),
                size: content.len() as i64,
                filename: "model.bin".to_string(),
                location: String::new(),
                mimetype: String::new(),
            },
            architecture: None,
            sourceLanguage: "en".to_string(),
            targetLanguage: "de".to_string(),
            decompressed_hash: Some(hash),
            decompressed_size: Some(content.len() as i64),
            id: "1".to_string(),
            last_modified: 0,
        };

        let data = super::RecordsData {
            data: vec![item],
            ..Default::default()
        };
        super::set_records(data);

        let downloader = crate::models::download::Downloader::new();
        let downloaded = super::download_model_to_dir(
            &downloader,
            &target_dir,
            "en",
            "de",
            None,
            Some(&src_dir),
        )
        .await
        .unwrap();

        assert_eq!(downloaded.get("model").map(String::as_str), Some("1.0"));
        assert!(target_dir.join("model.bin").exists());
        assert_eq!(tokio::fs::read(target_dir.join("model.bin")).await.unwrap(), content);

        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }
}
