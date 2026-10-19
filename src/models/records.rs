//! Model records: fetch from Mozilla Firefox Translations service, cache locally.
//!
//! Records are fetched from `firefox.settings.services.mozilla.com` and describe
//! available models, their versions, and download URLs.

use std::collections::{HashMap, HashSet};
use std::path::Path;

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

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct RecordsData {
    pub data: Vec<RecordItem>,
    #[serde(skip)]
    pub pairs_index: HashMap<String, HashSet<String>>,
    #[serde(skip)]
    pub supported_languages: HashSet<String>,
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

        self.pairs_index = pairs_index;
        self.supported_languages = supported_languages;
    }
}
/// Installed model records, swapped atomically so the background updater can
/// refresh the table while request threads take a consistent snapshot.
static GLOBAL_RECORDS: std::sync::RwLock<Option<std::sync::Arc<RecordsData>>> =
    std::sync::RwLock::new(None);

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
    *records_lock().write().unwrap_or_else(|e| e.into_inner()) = Some(std::sync::Arc::new(records));
}

fn with_records<R>(f: impl FnOnce(&RecordsData) -> R) -> Option<R> {
    records_lock()
        .read()
        .ok()
        .and_then(|guard| guard.as_deref().map(f))
}

fn records() -> Option<std::sync::Arc<RecordsData>> {
    records_lock().read().ok().and_then(|guard| guard.clone())
}

fn records_lock() -> &'static std::sync::RwLock<Option<std::sync::Arc<RecordsData>>> {
    &GLOBAL_RECORDS
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
    records().map_or(vec![], |r| {
        let mut pairs: Vec<String> = Vec::new();
        for (from, targets) in &r.pairs_index {
            for to in targets {
                pairs.push(format!("{}_{}", from, to));
            }
        }
        pairs.sort();
        pairs
    })
}

/// Get all supported language codes.
pub fn get_supported_languages() -> Vec<String> {
    records().map_or(vec![], |r| {
        let mut langs: Vec<String> = r.supported_languages.iter().cloned().collect();
        langs.sort();
        langs
    })
}

/// Check if a language code has at least one model record (source or target).
pub fn is_supported_language(code: &str) -> bool {
    let code = crate::utils::lang::normalize_language_code_cow(code);
    with_records(|r| r.supported_languages.contains(code.as_ref())).unwrap_or(true)
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
    let lang_pair_dir = cfg.model_dir.join(format!("{}_{}", from_lang, to_lang));
    tokio::fs::create_dir_all(&lang_pair_dir)
        .await
        .map_err(|e| format!("Failed to create model dir: {}", e))?;

    let downloader = Downloader::new();
    let downloaded =
        download_model_to_dir(&downloader, &lang_pair_dir, from_lang, to_lang, version).await?;

    // Persist version info for the updater
    write_installed_version(&lang_pair_dir, &downloaded).await?;

    Ok(())
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
    lang_pair_dir: &Path,
    from_lang: &str,
    to_lang: &str,
    version: Option<&str>,
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

    tokio::fs::create_dir_all(lang_pair_dir)
        .await
        .map_err(|e| format!("Failed to create model dir: {}", e))?;

    let mut downloaded_versions: HashMap<String, String> = HashMap::new();

    for &record in &matched {
        let filename = &record.attachment.filename;
        let file_url = format!("{}/{}", ATTACHMENTS_BASE_URL, record.attachment.location);
        let compressed_path = lang_pair_dir.join(filename);
        let decompressed_filename = filename.trim_end_matches(".zst");
        let decompressed_path = lang_pair_dir.join(decompressed_filename);

        let needs_download = if decompressed_path.exists() {
            match &record.decompressed_hash {
                Some(expected_hash) => {
                    compute_file_hash(&decompressed_path).await? != *expected_hash
                }
                // No hash to check: trust the existing file (same as before).
                None => false,
            }
        } else {
            true
        };

        if !needs_download {
            downloaded_versions.insert(record.file_type.clone(), record.version.clone());
            continue;
        }

        downloader.download(&file_url, &compressed_path).await?;

        if filename.ends_with(".zst") {
            if let Err(e) = downloader
                .decompress(&compressed_path, &decompressed_path)
                .await
            {
                let _ = tokio::fs::remove_file(&compressed_path).await;
                let _ = tokio::fs::remove_file(&decompressed_path).await;
                return Err(e);
            }
            let _ = tokio::fs::remove_file(&compressed_path).await;
        }
        // Verify the produced file before registering its version; otherwise a
        // corrupt download would be pinned as "latest" and never re-fetched.
        if let Some(expected_hash) = &record.decompressed_hash {
            let local_hash = compute_file_hash(&decompressed_path).await?;
            if local_hash != *expected_hash {
                let _ = tokio::fs::remove_file(&decompressed_path).await;
                let _ = tokio::fs::remove_file(&compressed_path).await;
                return Err(format!(
                    "Hash mismatch for {} (expected {}, got {})",
                    decompressed_filename, expected_hash, local_hash
                ));
            }
        }

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

async fn compute_file_hash(path: &Path) -> Result<String, String> {
    use sha2::Digest;
    use tokio::io::AsyncReadExt;

    // Stream in chunks instead of loading the whole (tens-of-MB) file.
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| format!("Failed to open file for hash {}: {e}", path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .await
            .map_err(|e| format!("Failed to read file for hash {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash = hasher.finalize();
    Ok(hash.iter().map(|b| format!("{:02x}", b)).collect())
}

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
    #[tokio::test]
    async fn test_init_records_behavior() {
        // 1. When local records.json exists, it should load cleanly
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
}
