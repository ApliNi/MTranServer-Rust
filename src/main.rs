#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use mtranserver_lib::config::Cli;
use mtranserver_lib::{api, config, engine, logger, models, service, utils};

use std::process;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let languages = cli.languages;
    let download = cli.download.clone();
    let bench_qps = cli.bench_qps.clone();

    // Initialize config first (subcommands and init_records both need it)
    config::init(cli);

    if languages {
        list_languages().await;
        return;
    }

    if let Some(pair) = bench_qps {
        bench_qps::run(&pair).await;
        return;
    }

    if !download.is_empty() {
        // "all" expands to every language pair in the records; mixed forms
        // like `--download all en_zh-Hans` download everything.
        let pairs = if download
            .iter()
            .any(|t| t.split([',', ' ']).any(|s| s.eq_ignore_ascii_case("all")))
        {
            if let Err(e) = models::init_records().await {
                eprintln!("Failed to initialize records: {}", e);
                return;
            }
            models::get_language_pairs()
        } else {
            download
        };
        download_models(&pairs).await;
        return;
    }

    logger::init();
    utils::profiling::set_enabled(config::get().profile);
    service::init();
    crate::utils::cache::init();
    engine::rules::init_rules_from_config();

    tracing::info!("Initializing MTranServer...");
    tracing::info!(
        "hardware acceleration: {}",
        crate::utils::arch::acceleration_report()
    );

    if let Err(e) = models::init_records().await {
        tracing::error!("Failed to initialize model records: {}", e);
        // Run the same cleanup as the graceful-shutdown path before exiting.
        service::cleanup_all_engines();
        logger::shutdown();
        process::exit(1);
    }

    if let Err(e) = server::run().await {
        tracing::error!("Server error: {}", e);
        service::cleanup_all_engines();
        logger::shutdown();
        process::exit(1);
    }
}

mod server {
    use std::net::SocketAddr;

    pub async fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cfg = crate::config::get();
        let app = crate::api::create_router(cfg);

        let host = if cfg.host.contains(':') && !cfg.host.starts_with('[') {
            format!("[{}]", cfg.host)
        } else {
            cfg.host.clone()
        };
        let addr: SocketAddr = format!("{}:{}", host, cfg.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        tracing::info!("Listening on {}", addr);
        let listener = tokio::net::TcpListener::bind(addr).await?;

        // Listen for SIGINT/SIGTERM for graceful shutdown
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();

        tracing::info!("Server started.");
        // Warm up --keep-models pairs in the background so pinned models are
        // resident before the first request arrives.
        crate::service::preload_engines();

        // Background model updater: checks installed models for newer versions
        // and hot-swaps them without interrupting service. The handle is kept
        // so graceful shutdown can cancel the task instead of leaving it
        // running until process exit.
        let updater_handle =
            crate::models::updater::spawn(cfg.config_dir.clone(), cfg.model_dir.clone());

        let graceful_shutdown = async move {
            // Wait for shutdown signal
            tokio::select! {
                _ = ctrl_c => {
                    tracing::info!("Received SIGINT, shutting down gracefully...");
                }
                _ = async {
                    #[cfg(unix)]
                    if let Some(sig) = &mut sigterm {
                        sig.recv().await;
                    }
                    #[cfg(not(unix))]
                    std::future::pending::<()>().await;
                } => {
                    tracing::info!("Received SIGTERM, shutting down gracefully...");
                }
            }
        };

        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(graceful_shutdown)
        .await?;

        // Stop the background updater before tearing down engines so it cannot
        // race with cleanup.
        updater_handle.abort();

        crate::utils::profiling::dump();
        crate::service::cleanup_all_engines();
        tracing::info!("Shutdown complete.");
        crate::logger::shutdown();

        Ok(())
    }
}

async fn list_languages() {
    // Subcommands run before logger::init(), so report errors directly on
    // stderr instead of through the (uninitialized) tracing pipeline.
    if let Err(e) = models::init_records().await {
        eprintln!("Failed to initialize records: {}", e);
        return;
    }

    // Format 1: language pairs, e.g. "en_zh-Hans zh-Hans_en ..."
    let pairs = models::get_language_pairs();
    println!("Language pairs:");
    println!("{}", pairs.join(" "));
    println!();

    // Format 2: deduplicated single languages, e.g. "ar az be bg ..."
    let languages = models::get_supported_languages();
    println!("Languages:");
    println!("{}", languages.join(" "));
}

/// Split a single language-pair token ("en_zh-Hans" or "en-zh-Hans") into
/// normalized (from, to). `None` for malformed tokens.
fn parse_language_pair(token: &str) -> Option<(String, String)> {
    let (from, to) = crate::utils::lang::split_language_pair(token)?;
    Some((
        crate::utils::lang::normalize_language_code(from),
        crate::utils::lang::normalize_language_code(to),
    ))
}

async fn download_models(pairs: &[String]) {
    if let Err(e) = models::init_records().await {
        eprintln!("Failed to initialize records: {}", e);
        return;
    }

    for pair_str in pairs {
        // Allow comma/space-separated tokens inside a single argument (defence
        // in depth — clap's value_delimiter already handles the CLI path).
        for token in pair_str.split([',', ' ']).filter(|s| !s.is_empty()) {
            let Some((from, to)) = parse_language_pair(token) else {
                eprintln!("Invalid language pair: {}", token);
                continue;
            };
            match models::download_model(&from, &to, None).await {
                Ok(_) => println!("Downloaded/verified model for {} -> {}", from, to),
                Err(e) => eprintln!("Failed to download model for {} -> {}: {}", from, to, e),
            }
        }
    }
}

/// Built-in maximum-QPS benchmark (`--bench-qps FROM_TO`).
///
/// Measures the translation throughput this host can sustain: every request
/// carries a fixed `--max-sentence-length`-character input, the result cache
/// is bypassed so each call performs the full forward pass, and the load runs
/// on `--max-threads` concurrent workers (0 = all logical cores). Prints
/// wall/CPU time, QPS and per-request latency percentiles, then exits.
mod bench_qps {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Benchmark duration per measurement phase.
    const DURATION: Duration = Duration::from_secs(10);

    pub async fn run(pair_token: &str) {
        let Some((from, to)) = super::parse_language_pair(pair_token) else {
            eprintln!(
                "Invalid language pair: {} (expected e.g. en_zh-Hans)",
                pair_token
            );
            return;
        };

        super::logger::init();
        super::service::init();
        crate::utils::cache::init();
        crate::engine::rules::init_rules_from_config();
        if let Err(e) = super::models::init_records().await {
            eprintln!("Failed to initialize records: {}", e);
            super::service::cleanup_all_engines();
            super::logger::shutdown();
            std::process::exit(1);
        }

        let cfg = super::config::get();
        let workers = if cfg.max_threads > 0 {
            cfg.max_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .max(1)
        };

        // Fixed-length input: exactly `bench_qps_input_length` characters of
        // realistic English prose, repeated and truncated to the limit.
        let input_len = cfg.bench_qps_input_length.max(1);
        let unit = "The quick brown fox jumps over the lazy dog near the river bank while ";
        let repeats = input_len.div_ceil(unit.chars().count());
        let text: String = unit.repeat(repeats).chars().take(input_len).collect();

        println!("[bench-qps] pair: {} -> {}", from, to);
        println!("[bench-qps] input length: {} chars", text.chars().count());
        println!(
            "[bench-qps] workers: {} (max_threads={})",
            workers, cfg.max_threads
        );

        // Warm up: load the model and verify the pair is usable.
        match super::service::translate_uncached(&from, &to, "Hello.").await {
            Ok(_) => {}
            Err(e) => {
                eprintln!("[bench-qps] model unavailable: {}", e);
                super::service::cleanup_all_engines();
                super::logger::shutdown();
                std::process::exit(1);
            }
        }

        println!("[bench-qps] running for {}s ...", DURATION.as_secs());

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completions = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(AtomicU64::new(0));
        // Per-request wall latencies in nanoseconds, collected lock-free and
        // post-processed once at the end.
        let latencies: Arc<parking_lot::Mutex<Vec<u64>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let start = Instant::now();

        let tasks: Vec<_> = (0..workers)
            .map(|_| {
                let stop = stop.clone();
                let completions = completions.clone();
                let errors = errors.clone();
                let latencies = latencies.clone();
                let from = from.clone();
                let to = to.clone();
                let text = text.clone();
                tokio::spawn(async move {
                    while !stop.load(Ordering::Relaxed) {
                        let t0 = Instant::now();
                        match super::service::translate_uncached(&from, &to, &text).await {
                            Ok(_) => {
                                completions.fetch_add(1, Ordering::Relaxed);
                                latencies.lock().push(t0.elapsed().as_nanos() as u64);
                            }
                            Err(_) => {
                                errors.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                })
            })
            .collect();

        tokio::time::sleep(DURATION).await;
        stop.store(true, Ordering::Relaxed);
        for t in tasks {
            let _ = t.await;
        }

        let wall = start.elapsed();
        let total = completions.load(Ordering::Relaxed);
        let failed = errors.load(Ordering::Relaxed);
        let qps = total as f64 / wall.as_secs_f64();

        let mut lats = latencies.lock().clone();
        lats.sort_unstable();
        let pct = |p: f64| -> f64 {
            if lats.is_empty() {
                return 0.0;
            }
            let idx = ((lats.len() as f64) * p).min((lats.len() - 1) as f64) as usize;
            lats[idx] as f64 / 1_000_000.0
        };

        println!();
        println!("==== QPS Benchmark Result ====");
        println!("pair            : {} -> {}", from, to);
        println!("input length    : {} chars", text.chars().count());
        println!("workers         : {}", workers);
        println!("duration        : {:.2}s", wall.as_secs_f64());
        println!("requests        : {} ok, {} failed", total, failed);
        println!("QPS             : {:.1}", qps);
        println!("latency p50     : {:.2} ms", pct(0.50));
        println!("latency p95     : {:.2} ms", pct(0.95));
        println!("latency p99     : {:.2} ms", pct(0.99));

        super::service::cleanup_all_engines();
        super::logger::shutdown();
    }
}
