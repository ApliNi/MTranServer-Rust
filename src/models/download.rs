//! HTTP download with proxy support and zstd decompression.
//!
//! Follows HTTP_PROXY / HTTPS_PROXY environment variables,
//! and respects SOCKS5 proxies via the `socks` feature of reqwest.

use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use reqwest::Client;

/// Shared download client so repeated update cycles reuse the connection
/// pool and TLS sessions instead of rebuilding a `reqwest::Client` per run.
static SHARED_CLIENT: OnceLock<Client> = OnceLock::new();

fn shared_client() -> &'static Client {
    SHARED_CLIENT.get_or_init(|| build_client().expect("Failed to build HTTP client"))
}

fn build_client() -> Result<Client, String> {
    let mut builder = Client::builder()
        .timeout(Duration::from_secs(1800))
        .user_agent(concat!("MTranServer/", env!("CARGO_PKG_VERSION")));

    // Check for proxy env vars (HTTPS first, then HTTP).
    let proxy = std::env::var("HTTPS_PROXY")
        .or_else(|_| std::env::var("https_proxy"))
        .or_else(|_| std::env::var("HTTP_PROXY"))
        .or_else(|_| std::env::var("http_proxy"));

    if let Ok(proxy_url) = proxy {
        if !proxy_url.is_empty() {
            if let Ok(p) = reqwest::Proxy::all(&proxy_url) {
                builder = builder.proxy(p);
            }
        }
    }

    builder
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))
}

/// Buffer capacity for stream download and decompression (128 KB).
const STREAM_BUFFER_CAPACITY: usize = 128 * 1024;
const DECOMPRESS_BUFFER_CAPACITY: usize = STREAM_BUFFER_CAPACITY;

/// Downloader with configurable timeout and proxy support.
pub struct Downloader {
    client: Client,
}

impl Downloader {
    /// Create a new downloader backed by the process-wide shared client.
    ///
    /// Automatically detects system proxy settings from:
    /// - `HTTP_PROXY` / `http_proxy`
    /// - `HTTPS_PROXY` / `https_proxy`
    pub fn new() -> Self {
        Self {
            client: shared_client().clone(),
        }
    }

    #[inline]
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Download a file from URL to a local path.
    pub async fn download(&self, url: &str, output_path: &Path) -> Result<(), String> {
        self.download_verified(url, output_path, None, None).await
    }

    /// Download a file from URL to a local path with optional size and SHA-256 verification.
    ///
    /// The payload is written to `<path>.part` first and atomically renamed into place.
    /// If the destination file already exists, it is verified against `expected_size` and
    /// `expected_sha256`; if valid, download is skipped. If corrupted or truncated, it is
    /// re-downloaded.
    pub async fn download_verified(
        &self,
        url: &str,
        output_path: &Path,
        expected_size: Option<u64>,
        expected_sha256: Option<&str>,
    ) -> Result<(), String> {
        if output_path.exists() {
            let mut valid = true;
            if let Some(expected_len) = expected_size {
                match tokio::fs::metadata(output_path).await {
                    Ok(metadata) => {
                        if metadata.len() != expected_len {
                            valid = false;
                        }
                    }
                    Err(_) => {
                        valid = false;
                    }
                }
            }
            if valid {
                if let Some(expected_hash) = expected_sha256 {
                    if !expected_hash.is_empty() {
                        match compute_file_hash(output_path).await {
                            Ok(h) if h == expected_hash => {}
                            _ => {
                                valid = false;
                            }
                        }
                    }
                }
            }
            if valid {
                return Ok(());
            }
            let _ = tokio::fs::remove_file(output_path).await;
        }

        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let mut response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!(
                "HTTP {}: {}",
                response.status().as_u16(),
                response.status().canonical_reason().unwrap_or("Unknown")
            ));
        }

        let part_path = part_path_for(output_path)?;
        let file = tokio::fs::File::create(&part_path)
            .await
            .map_err(|e| format!("Failed to create temporary file {}: {e}", part_path.display()))?;
        let mut writer = tokio::io::BufWriter::with_capacity(STREAM_BUFFER_CAPACITY, file);

        let stream_res: Result<(), String> = async {
            use tokio::io::AsyncWriteExt;
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|e| format!("Failed reading download stream: {e}"))?
            {
                writer
                    .write_all(&chunk)
                    .await
                    .map_err(|e| format!("Failed to write {}: {e}", part_path.display()))?;
            }
            writer
                .flush()
                .await
                .map_err(|e| format!("Failed to flush {}: {e}", part_path.display()))?;
            Ok(())
        }
        .await;

        drop(writer);

        if let Err(err) = stream_res {
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(err);
        }

        if let Some(expected_len) = expected_size {
            if let Ok(metadata) = tokio::fs::metadata(&part_path).await {
                if metadata.len() != expected_len {
                    let _ = tokio::fs::remove_file(&part_path).await;
                    return Err(format!(
                        "Downloaded file size mismatch for {} (expected {}, got {})",
                        output_path.display(),
                        expected_len,
                        metadata.len()
                    ));
                }
            }
        }

        if let Some(expected_hash) = expected_sha256 {
            if !expected_hash.is_empty() {
                let actual_hash = compute_file_hash(&part_path).await?;
                if actual_hash != expected_hash {
                    let _ = tokio::fs::remove_file(&part_path).await;
                    return Err(format!(
                        "Downloaded file hash mismatch for {} (expected {}, got {})",
                        output_path.display(),
                        expected_hash,
                        actual_hash
                    ));
                }
            }
        }

        rename_atomic(&part_path, output_path).await
    }

    /// Decompress a zstd file to the output path using streaming decompression.
    ///
    /// Streams chunks from the input file through [`zstd::stream::copy_decode`]
    /// into a sibling `.part` file inside [`tokio::task::spawn_blocking`], avoiding
    /// loading the whole uncompressed model into memory. The file is flushed and
    /// atomically renamed into place.
    pub async fn decompress(&self, input_path: &Path, output_path: &Path) -> Result<(), String> {
        let input_path = input_path.to_path_buf();
        let output_path = output_path.to_path_buf();

        tokio::task::spawn_blocking(move || decompress_blocking(&input_path, &output_path))
            .await
            .map_err(|e| format!("Decompression task failed: {e}"))?
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}

/// Write `bytes` to a sibling `.part` file, then rename it over `final_path`.
///
/// On Unix the rename is atomic; on Windows a plain rename cannot overwrite,
/// so remove the destination first — the window without the file is the same
/// one the updater already tolerates, but the file content itself can never
/// be observed half-written.
fn part_path_for(final_path: &Path) -> Result<PathBuf, String> {
    let name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| format!("Invalid target path: {}", final_path.display()))?;
    Ok(final_path.with_file_name(format!("{name}.part")))
}

pub(crate) fn rename_atomic_sync(part_path: &Path, final_path: &Path) -> Result<(), String> {
    #[cfg(not(windows))]
    {
        std::fs::rename(part_path, final_path).map_err(|e| {
            let _ = std::fs::remove_file(part_path);
            format!(
                "Failed to move {} into place at {}: {e}",
                part_path.display(),
                final_path.display()
            )
        })
    }
    #[cfg(windows)]
    {
        if final_path.exists() {
            let bak = {
                let mut p = final_path.as_os_str().to_owned();
                p.push(".bak");
                std::path::PathBuf::from(p)
            };
            let _ = std::fs::remove_file(&bak);
            let mut renamed_to_bak = false;
            let mut last_err = None;
            for _ in 0..5 {
                match std::fs::rename(final_path, &bak) {
                    Ok(()) => {
                        renamed_to_bak = true;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
            if !renamed_to_bak {
                return Err(format!(
                    "Failed to move old {} out of place: {}",
                    final_path.display(),
                    last_err.map(|e| e.to_string()).unwrap_or_default()
                ));
            }

            let mut renamed_part = false;
            let mut part_err = None;
            for _ in 0..5 {
                match std::fs::rename(part_path, final_path) {
                    Ok(()) => {
                        renamed_part = true;
                        break;
                    }
                    Err(e) => {
                        part_err = Some(e);
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }

            if !renamed_part {
                let mut restored = false;
                for _ in 0..5 {
                    if std::fs::rename(&bak, final_path).is_ok() {
                        restored = true;
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                if !restored {
                    tracing::error!(
                        "Failed to restore {} from .bak after failed swap",
                        final_path.display()
                    );
                }
                let _ = std::fs::remove_file(part_path);
                return Err(format!(
                    "Failed to move {} into place at {}: {}",
                    part_path.display(),
                    final_path.display(),
                    part_err.map(|e| e.to_string()).unwrap_or_default()
                ));
            }

            let _ = std::fs::remove_file(&bak);
            Ok(())
        } else {
            std::fs::rename(part_path, final_path).map_err(|e| {
                let _ = std::fs::remove_file(part_path);
                format!(
                    "Failed to move {} into place at {}: {e}",
                    part_path.display(),
                    final_path.display()
                )
            })
        }
    }
}

pub(crate) async fn rename_atomic(part_path: &Path, final_path: &Path) -> Result<(), String> {
    #[cfg(not(windows))]
    {
        tokio::fs::rename(part_path, final_path).await.map_err(|e| {
            let _ = tokio::fs::remove_file(part_path);
            format!(
                "Failed to move {} into place at {}: {e}",
                part_path.display(),
                final_path.display()
            )
        })
    }
    #[cfg(windows)]
    {
        if final_path.exists() {
            let bak = {
                let mut p = final_path.as_os_str().to_owned();
                p.push(".bak");
                std::path::PathBuf::from(p)
            };
            let _ = tokio::fs::remove_file(&bak).await;
            let mut renamed_to_bak = false;
            let mut last_err = None;
            for _ in 0..5 {
                match tokio::fs::rename(final_path, &bak).await {
                    Ok(()) => {
                        renamed_to_bak = true;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                }
            }
            if !renamed_to_bak {
                return Err(format!(
                    "Failed to move old {} out of place: {}",
                    final_path.display(),
                    last_err.map(|e| e.to_string()).unwrap_or_default()
                ));
            }

            let mut renamed_part = false;
            let mut part_err = None;
            for _ in 0..5 {
                match tokio::fs::rename(part_path, final_path).await {
                    Ok(()) => {
                        renamed_part = true;
                        break;
                    }
                    Err(e) => {
                        part_err = Some(e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                }
            }

            if !renamed_part {
                let mut restored = false;
                for _ in 0..5 {
                    if tokio::fs::rename(&bak, final_path).await.is_ok() {
                        restored = true;
                        break;
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                }
                if !restored {
                    tracing::error!(
                        "Failed to restore {} from .bak after failed swap",
                        final_path.display()
                    );
                }
                let _ = tokio::fs::remove_file(part_path).await;
                return Err(format!(
                    "Failed to move {} into place at {}: {}",
                    part_path.display(),
                    final_path.display(),
                    part_err.map(|e| e.to_string()).unwrap_or_default()
                ));
            }

            let _ = tokio::fs::remove_file(&bak).await;
            Ok(())
        } else {
            tokio::fs::rename(part_path, final_path).await.map_err(|e| {
                let _ = std::fs::remove_file(part_path);
                format!(
                    "Failed to move {} into place at {}: {e}",
                    part_path.display(),
                    final_path.display()
                )
            })
        }
    }
}

pub(crate) async fn compute_file_hash(path: &Path) -> Result<String, String> {
    use sha2::Digest;
    use tokio::io::AsyncReadExt;

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

fn decompress_blocking(input_path: &Path, output_path: &Path) -> Result<(), String> {
    let part_path = part_path_for(output_path)?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {e}", parent.display()))?;
    }

    let input_file = std::fs::File::open(input_path).map_err(|e| {
        format!(
            "Failed to open compressed file {}: {e}",
            input_path.display()
        )
    })?;
    let reader = BufReader::with_capacity(DECOMPRESS_BUFFER_CAPACITY, input_file);

    let part_file = std::fs::File::create(&part_path).map_err(|e| {
        format!(
            "Failed to create temporary file {}: {e}",
            part_path.display()
        )
    })?;
    let mut writer = BufWriter::with_capacity(DECOMPRESS_BUFFER_CAPACITY, part_file);

    let decode_result = zstd::stream::copy_decode(reader, &mut writer)
        .map_err(|e| format!("Zstd decompression failed: {e}"));

    let flush_result = if decode_result.is_ok() {
        writer
            .flush()
            .map_err(|e| format!("Failed to flush {}: {e}", part_path.display()))
    } else {
        Ok(())
    };

    // Ensure writer is dropped before renaming, closing the open handle on Windows.
    drop(writer);

    if let Err(err) = decode_result {
        let _ = std::fs::remove_file(&part_path);
        return Err(err);
    }

    if let Err(err) = flush_result {
        let _ = std::fs::remove_file(&part_path);
        return Err(err);
    }

    rename_atomic_sync(&part_path, output_path)
}

pub(crate) async fn write_atomic(final_path: &Path, bytes: &[u8]) -> Result<(), String> {
    let part_path = part_path_for(final_path)?;

    tokio::fs::write(&part_path, bytes)
        .await
        .map_err(|e| format!("Failed to write {}: {e}", part_path.display()))?;

    rename_atomic(&part_path, final_path).await
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "mtran_test_download_{name}_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn test_decompress_stream_basic() {
        let temp = TempDirGuard::new("basic");
        let input_path = temp.path().join("model.bin.zst");
        let output_path = temp.path().join("model.bin");

        let original_data = b"Hello streaming zstd decompression! Repeated pattern: ".repeat(1000);
        let compressed = zstd::encode_all(&original_data[..], 3).unwrap();
        std::fs::write(&input_path, &compressed).unwrap();

        let downloader = Downloader::new();
        downloader
            .decompress(&input_path, &output_path)
            .await
            .unwrap();

        assert!(output_path.exists());
        let decompressed = std::fs::read(&output_path).unwrap();
        assert_eq!(decompressed, original_data);

        let part_path = temp.path().join("model.bin.part");
        assert!(!part_path.exists());
    }

    #[tokio::test]
    async fn test_decompress_stream_multi_chunk() {
        let temp = TempDirGuard::new("multi_chunk");
        let input_path = temp.path().join("large.bin.zst");
        let output_path = temp.path().join("large.bin");

        let mut original_data = Vec::with_capacity(512 * 1024);
        for i in 0..(512 * 1024) {
            original_data.push((i % 251) as u8);
        }
        let compressed = zstd::encode_all(&original_data[..], 3).unwrap();
        std::fs::write(&input_path, &compressed).unwrap();

        let downloader = Downloader::new();
        downloader
            .decompress(&input_path, &output_path)
            .await
            .unwrap();

        assert!(output_path.exists());
        let decompressed = std::fs::read(&output_path).unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[tokio::test]
    async fn test_decompress_overwrite_existing() {
        let temp = TempDirGuard::new("overwrite");
        let input_path = temp.path().join("model.bin.zst");
        let output_path = temp.path().join("model.bin");

        std::fs::write(&output_path, b"old obsolete data").unwrap();

        let new_data = b"new decompressed content".to_vec();
        let compressed = zstd::encode_all(&new_data[..], 3).unwrap();
        std::fs::write(&input_path, &compressed).unwrap();

        let downloader = Downloader::new();
        downloader
            .decompress(&input_path, &output_path)
            .await
            .unwrap();

        let decompressed = std::fs::read(&output_path).unwrap();
        assert_eq!(decompressed, new_data);
    }

    #[tokio::test]
    async fn test_decompress_invalid_input_cleans_up_part() {
        let temp = TempDirGuard::new("invalid_input");
        let input_path = temp.path().join("corrupted.bin.zst");
        let output_path = temp.path().join("target.bin");
        let part_path = temp.path().join("target.bin.part");

        std::fs::write(&input_path, b"this is definitely not a valid zstd stream").unwrap();

        let downloader = Downloader::new();
        let result = downloader.decompress(&input_path, &output_path).await;
        assert!(result.is_err());
        assert!(!output_path.exists());
        assert!(!part_path.exists());
    }

    #[tokio::test]
    async fn test_decompress_preserves_existing_on_error() {
        let temp = TempDirGuard::new("preserve_existing");
        let input_path = temp.path().join("corrupted.bin.zst");
        let output_path = temp.path().join("target.bin");

        let existing_data = b"do not overwrite me if decompression fails";
        std::fs::write(&output_path, existing_data).unwrap();
        std::fs::write(&input_path, b"invalid zstd data").unwrap();

        let downloader = Downloader::new();
        let result = downloader.decompress(&input_path, &output_path).await;
        assert!(result.is_err());

        assert_eq!(std::fs::read(&output_path).unwrap(), existing_data);
    }

    #[tokio::test]
    async fn test_write_atomic_basic() {
        let temp = TempDirGuard::new("write_atomic");
        let target_path = temp.path().join("atomic.txt");
        let data = b"atomic write verification";

        write_atomic(&target_path, data).await.unwrap();
        assert_eq!(std::fs::read(&target_path).unwrap(), data);

        let new_data = b"atomic write overwrite";
        write_atomic(&target_path, new_data).await.unwrap();
        assert_eq!(std::fs::read(&target_path).unwrap(), new_data);
    }
}
