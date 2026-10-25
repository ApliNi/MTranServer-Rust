//! Background model updater that periodically checks for newer versions of
//! installed models and hot-swaps them without interrupting service.
//!
//! The updater runs as a `tokio` task spawned after the server starts. It
//! performs one check immediately and then repeats every `interval` hours.
//!
//! # Safety
//!
//! - Engines map model files using `memmap2::Mmap`. Before directory replacement,
//!   running engines are evicted and the updater waits for active requests to finish,
//!   ensuring all memory-mapped file handles are closed before swapping on Windows.
//! - The updater downloads to an isolated staging directory, verifies hashes, and
//!   only then atomically swaps the new directory into place.
//! - After eviction and swap, the next request to that pair loads the new model automatically.
use std::path::{Path, PathBuf};

use crate::config;
use crate::models::records;

/// Start the background model updater.
///
/// Spawns a non-blocking task that:
/// 1. Runs once shortly after the server starts.
/// 2. Repeats at the configured interval.
///
/// Call this from `server::run()` after the listener is up. The returned
/// handle should be `abort()`ed on graceful shutdown so the task cannot
/// outlive the server.
pub fn spawn(config_dir: PathBuf, model_dir: PathBuf) -> tokio::task::JoinHandle<()> {
    let cfg = config::get();
    if !cfg.check_model_update || cfg.enable_offline_mode {
        // Disabled: return an already-finished handle so callers can treat
        // every path uniformly.
        return tokio::task::spawn(async {});
    }
    let interval_secs = cfg.model_update_interval_secs;
    tokio::spawn(async move {
        // Brief delay so startup logging is not interleaved.
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        // Always run one check (bare `--check-model-update` stops here).
        if let Err(e) = run_once(&config_dir, &model_dir).await {
            tracing::warn!("Model update check failed: {}", e);
        }

        if interval_secs == 0 {
            return; // one-shot mode
        }

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            if let Err(e) = run_once(&config_dir, &model_dir).await {
                tracing::warn!("Model update check failed: {}", e);
            }
        }
    })
}

/// One complete update cycle: fetch records, check each installed pair, and
/// download + swap any out-of-date files.
async fn run_once(config_dir: &Path, model_dir: &Path) -> Result<(), String> {
    tracing::info!("Checking for model records and model updates...");
    // Refresh the record table so we have the latest version metadata.
    records::fetch_records(config_dir).await?;

    let installed = records::installed_pairs(model_dir).await;
    for (from, to) in &installed {
        check_and_update(model_dir, from, to).await;
    }
    tracing::info!("Model update check completed.");

    Ok(())
}

/// Check one language pair for an available update and apply it if found.
async fn check_and_update(model_dir: &Path, from_lang: &str, to_lang: &str) {
    let pair_dir = model_dir.join(format!("{}_{}", from_lang, to_lang));

    // Read current installed versions (from version.json or inferred).
    let current = match records::read_installed_version(&pair_dir).await {
        Ok(Some(v)) => v,
        Ok(None) => match records::infer_installed_version(&pair_dir, from_lang, to_lang).await {
            Some(v) => v,
            None => {
                tracing::warn!(
                    "{} -> {}: cannot determine installed version, skipping",
                    from_lang,
                    to_lang
                );
                return;
            }
        },
        Err(e) => {
            tracing::warn!(
                "{} -> {}: failed to read version: {}",
                from_lang,
                to_lang,
                e
            );
            return;
        }
    };

    // Get the latest versions from the record table.
    let latest = match records::latest_record_versions(from_lang, to_lang) {
        Some(v) => v,
        None => {
            tracing::debug!("{} -> {}: no records found, skipping", from_lang, to_lang);
            return;
        }
    };

    // Check if any fileType has a newer version.
    let needs_update = latest.iter().any(|(ft, latest_ver)| {
        current.get(ft).map_or(true, |cur| {
            records::compare_versions(cur, latest_ver) == std::cmp::Ordering::Less
        })
    });

    if !needs_update {
        tracing::debug!("{} -> {}: already up to date", from_lang, to_lang);
        return;
    }

    tracing::info!(
        "{} -> {}: update available, downloading...",
        from_lang,
        to_lang
    );

    if let Err(e) = download_and_swap(model_dir, from_lang, to_lang).await {
        tracing::warn!("{} -> {}: update failed: {}", from_lang, to_lang, e);
    }
}

/// Download the latest model files to a staging directory, then atomically
/// swap the staging directory into place and evict the running engine.
async fn download_and_swap(model_dir: &Path, from_lang: &str, to_lang: &str) -> Result<(), String> {
    records::download_model_dir(model_dir, from_lang, to_lang, None).await?;
    tracing::info!("{} -> {}: atomically updated model", from_lang, to_lang);
    Ok(())
}
