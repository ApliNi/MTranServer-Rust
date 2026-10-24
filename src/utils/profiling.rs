//! Lightweight opt-in profiling counters for the translation hot path.
//!
//! Regions are timed with [`scope`]; the aggregate is printed by [`dump`],
//! which is a no-op until profiling is explicitly enabled via
//! [`set_enabled`] (wired to the `--profile` CLI flag).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::Instant;
use parking_lot::RwLock;

#[derive(Default)]
struct RegionStats {
    total_ns: AtomicU64,
    calls: AtomicU64,
}

static CELLS: LazyLock<RwLock<BTreeMap<&'static str, RegionStats>>> =
    LazyLock::new(|| RwLock::new(BTreeMap::new()));

fn record(name: &'static str, ns: u64) {
    {
        let map = CELLS.read();
        if let Some(stats) = map.get(name) {
            stats.total_ns.fetch_add(ns, Ordering::Relaxed);
            stats.calls.fetch_add(1, Ordering::Relaxed);
            return;
        }
    }

    let mut map = CELLS.write();
    let stats = map.entry(name).or_default();
    stats.total_ns.fetch_add(ns, Ordering::Relaxed);
    stats.calls.fetch_add(1, Ordering::Relaxed);
}

/// Accumulates the elapsed nanoseconds of the scoped region into `name` and
/// bumps the region's call count. When profiling is disabled the region runs
/// directly with only a cached flag check of overhead.
#[inline(always)]
pub fn scope<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let r = f();
    let ns = t0.elapsed().as_nanos() as u64;
    record(name, ns);
    r
}

/// Process-wide profiling switch, set once by [`set_enabled`].
fn enabled() -> bool {
    ON.load(Ordering::Relaxed) != 0
}

static ON: AtomicU64 = AtomicU64::new(0);

/// Enable or disable profiling. Called once at startup from the `--profile`
/// CLI flag; later calls change the behavior of subsequent [`scope`] regions.
pub fn set_enabled(on: bool) {
    ON.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Prints all counters (ns → ms) and resets them. No-op unless profiling was
/// enabled via [`set_enabled`].
pub fn dump() {
    if !enabled() {
        return;
    }
    let map = CELLS.write();
    println!("[prof] region                                   total_ms      calls  ms/call");
    let mut rows: Vec<(&'static str, u64, u64)> = Vec::new();
    for (&name, stats) in map.iter() {
        let ns = stats.total_ns.load(Ordering::Relaxed);
        let calls = stats.calls.load(Ordering::Relaxed);
        rows.push((name, ns, calls));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (name, ns, calls) in rows {
        let ms = ns as f64 / 1e6;
        let per = if calls > 0 {
            ms / calls as f64
        } else {
            f64::NAN
        };
        println!("[prof] {name:42} {ms:9.2} {calls:9} {per:9.4}");
    }
    // Reset after reporting.
    for stats in map.values() {
        stats.total_ns.store(0, Ordering::Relaxed);
        stats.calls.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling_lifecycle_and_concurrency() {
        set_enabled(false);
        assert!(!enabled());

        let res = scope("test_disabled", || 42);
        assert_eq!(res, 42);

        set_enabled(true);
        assert!(enabled());

        let res = scope("test_enabled", || 100);
        assert_eq!(res, 100);

        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..50 {
                        scope("test_concurrent", || {
                            std::hint::black_box(1 + 1);
                        });
                    }
                });
            }
        });

        {
            let map = CELLS.read();
            let stats = map.get("test_concurrent").unwrap();
            assert_eq!(stats.calls.load(Ordering::Relaxed), 400);
        }

        dump();

        {
            let map = CELLS.read();
            let stats = map.get("test_concurrent").unwrap();
            assert_eq!(stats.calls.load(Ordering::Relaxed), 0);
        }

        set_enabled(false);
    }
}
