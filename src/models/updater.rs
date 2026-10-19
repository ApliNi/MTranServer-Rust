//! Background model updater that periodically checks for newer versions of
//! installed models and hot-swaps them without interrupting service.
//!
//! The updater runs as a `tokio` task spawned after the server starts. It
//! performs one check immediately and then repeats every `interval` hours.
//!
//! # Safety
//!
//! - Engines load model data into memory (`std::fs::read`) and do not hold
//!   file handles, so replacing the on-disk files is safe for running engines.
//! - The updater downloads to a temporary directory under the pair's folder,
//!   verifies hashes, and only then moves the new files into place.
//! - After replacement the engine is evicted from the pool, so the next
//!   request to that pair loads the new model automatically.

use std::path::{Path, PathBuf};

use crate::config;
use crate::models::download::Downloader;
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

    if let Err(e) = download_and_swap(&pair_dir, from_lang, to_lang).await {
        tracing::warn!("{} -> {}: update failed: {}", from_lang, to_lang, e);
    }
}

/// Download the latest model files to a temporary directory, then move them
/// over the old files and evict the running engine.
async fn download_and_swap(pair_dir: &Path, from_lang: &str, to_lang: &str) -> Result<(), String> {
    let tmp_dir = pair_dir.join(".update");
    // Start from a clean temp dir (a previous interrupted run may have left files).
    if tmp_dir.exists() {
        tokio::fs::remove_dir_all(&tmp_dir)
            .await
            .map_err(|e| format!("Failed to clean temp dir: {}", e))?;
    }
    tokio::fs::create_dir_all(&tmp_dir)
        .await
        .map_err(|e| format!("Failed to create temp dir: {}", e))?;

    let downloader = Downloader::new();

    // Phase 1: download all files into the temp dir. Files whose hash already
    // matches are skipped, but since tmp_dir is empty this always downloads.
    let downloaded = records::download_model_to_dir(
        &downloader,
        &tmp_dir,
        from_lang,
        to_lang,
        None, // latest version
    )
    .await?;

    // Phase 2: every decompressed file should now be inside tmp_dir.
    // download_model_to_dir skips files whose hash already matches, but the
    // temp dir was freshly cleared, so every expected file was written.
    // Phase 2.5: evict the running engine before replacing files so that
    // Windows file locks from mmap and background readers are dropped.
    crate::service::evict_engine(from_lang, to_lang);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Phase 3: move the new files over the old ones. The decompressed
    // filenames in tmp_dir are the canonical model file names.
    let mut moved: Vec<String> = Vec::new();
    let mut entries = tokio::fs::read_dir(&tmp_dir)
        .await
        .map_err(|e| format!("Failed to read temp dir: {}", e))?;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry
            .file_type()
            .await
            .map(|t| t.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        let src = tmp_dir.join(&name);
        let dst = pair_dir.join(&name);
        move_into_place(&src, &dst, &name).await?;
        moved.push(name);
    }

    // Phase 4: persist the new version metadata.
    records::write_installed_version(pair_dir, &downloaded).await?;

    // Phase 5: ensure any racing re-loads are refreshed.
    crate::service::evict_engine(from_lang, to_lang);
    tracing::info!(
        "{} -> {}: updated {} files (swapped to v{:?})",
        from_lang,
        to_lang,
        moved.len(),
        downloaded
    );

    // Phase 6: clean up the temp dir.
    if tmp_dir.exists() {
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
    }

    Ok(())
}

/// Replace `dst` with `src`, tolerating an existing destination file.
///
/// On Unix `std::fs::rename` atomically replaces existing files, so there is
/// no window where the model directory lacks the file. Windows cannot rename
/// onto an existing open file, so instead of deleting the old file first
/// (which would leave a long no-file window while the multi-MB download is
/// renamed in), the old file is renamed to a `.bak` sibling — a fast metadata
/// operation that frees the destination name almost immediately — and only
/// then is the new file moved in. A crash between the two steps leaves the
/// `.bak` behind (the next swap cleans it up); the no-file window is a few
/// metadata operations wide, not a full download.
async fn move_into_place(src: &Path, dst: &Path, name: &str) -> Result<(), String> {
    #[cfg(unix)]
    {
        tokio::fs::rename(src, dst)
            .await
            .map_err(|e| format!("Failed to move {} into place: {}", name, e))
    }
    #[cfg(not(unix))]
    {
        let bak = {
            let mut p = dst.as_os_str().to_owned();
            p.push(".bak");
            std::path::PathBuf::from(p)
        };
        if dst.exists() {
            let _ = tokio::fs::remove_file(&bak).await;
            let mut renamed = false;
            let mut last_err = None;
            for _ in 0..5 {
                match tokio::fs::rename(dst, &bak).await {
                    Ok(()) => {
                        renamed = true;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                }
            }
            if !renamed {
                return Err(format!(
                    "Failed to move old {} out of place: {}",
                    name,
                    last_err.map(|e| e.to_string()).unwrap_or_default()
                ));
            }
        }
        if let Err(e) = tokio::fs::rename(src, dst).await {
            // Best-effort rollback so the pair directory keeps a usable model
            // even when the move-in fails.
            if !dst.exists() && tokio::fs::rename(&bak, dst).await.is_err() {
                tracing::error!(
                    "Failed to restore {} from .bak after failed swap: {}",
                    name,
                    e
                );
            }
            return Err(format!("Failed to move {} into place: {}", name, e));
        }
        let _ = tokio::fs::remove_file(&bak).await;
        Ok(())
    }
}
