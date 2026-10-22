use lru::LruCache;
use parking_lot::RwLock;

use crate::config;

/// 128-bit cache key combining two independent 64-bit hash streams to eliminate
/// accidental collisions and prevent cross-talk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey(pub u64, pub u64);

/// Shard count for the translation cache: power of two so shard selection is
/// a mask on the digest. Independent per-shard locks keep concurrent
/// requests from serialising on one global mutex.
const SHARDS: usize = 32;

#[repr(align(64))]
struct CacheShard(RwLock<Option<LruCache<CacheKey, std::sync::Arc<str>>>>);

static CACHE: [CacheShard; SHARDS] = [const { CacheShard(RwLock::new(None)) }; SHARDS];

/// True if the cache is enabled and configured with a positive size.
pub fn enabled() -> bool {
    config::is_initialized() && config::get().cache_size > 0
}

const SEED1: u64 = 0xcbf29ce484222325;
const SEED2: u64 = 0x84222325cbf29ce4;
const P1: u64 = 0x517cc1b727220a95;
const P2: u64 = 0x9e3779b97f4a7c15;
const SEPARATOR: u8 = 0xff;

#[inline(always)]
fn mix_step(h1: &mut u64, h2: &mut u64, v1: u64, v2: u64) {
    let m1 = (*h1 ^ v1) as u128 * P1 as u128;
    *h1 = (m1 as u64) ^ ((m1 >> 64) as u64) ^ v2;
    let m2 = (*h2 ^ v2) as u128 * P2 as u128;
    *h2 = (m2 as u64) ^ ((m2 >> 64) as u64) ^ v1;
}

#[inline(always)]
fn feed_bytes(h1: &mut u64, h2: &mut u64, bytes: &[u8]) {
    *h1 = (*h1 ^ (bytes.len() as u64)).wrapping_mul(P1);
    *h2 = (*h2 ^ (bytes.len() as u64)).wrapping_mul(P2);

    let mut chunks = bytes.chunks_exact(16);
    for chunk in &mut chunks {
        let v1 = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let v2 = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        mix_step(h1, h2, v1, v2);
    }

    let rem = chunks.remainder();
    if rem.len() >= 8 {
        let v1 = u64::from_le_bytes(rem[0..8].try_into().unwrap());
        let rem2 = &rem[8..];
        let mut v2 = 0u64;
        for (i, &b) in rem2.iter().enumerate() {
            v2 |= (b as u64) << (i * 8);
        }
        mix_step(h1, h2, v1, v2);
    } else if !rem.is_empty() {
        let mut v1 = 0u64;
        for (i, &b) in rem.iter().enumerate() {
            v1 |= (b as u64) << (i * 8);
        }
        mix_step(h1, h2, v1, 0);
    }

    *h1 = (*h1 ^ (SEPARATOR as u64)).wrapping_mul(P1);
    *h2 = (*h2 ^ (SEPARATOR as u64)).wrapping_mul(P2);
}

/// Compute a 128-bit collision-resistant cache key for request arguments.
///
/// Arguments are hashed using 16-byte dual wide-multiplication block streams,
/// separated by 0xFF delimiters to eliminate boundary ambiguity.
/// The resulting 128-bit state is finalized with splitmix64 avalanche transformations
/// to ensure uniform bit dispersion across both 64-bit lanes.
pub fn compute_key(args: &[&str]) -> CacheKey {
    let mut h1: u64 = SEED1;
    let mut h2: u64 = SEED2;

    for arg in args {
        feed_bytes(&mut h1, &mut h2, arg.as_bytes());
    }

    CacheKey(splitmix64(h1), splitmix64(h2))
}

/// Compute a 128-bit collision-resistant cache key incorporating an explicit rules version.
///
/// Hashes all request arguments and folds the language rules version into the dual block streams,
/// ensuring complete cache isolation between different versions of language rules.
pub fn compute_key_with_version(args: &[&str], version: u64) -> CacheKey {
    let mut h1: u64 = SEED1;
    let mut h2: u64 = SEED2;

    for arg in args {
        feed_bytes(&mut h1, &mut h2, arg.as_bytes());
    }

    feed_bytes(&mut h1, &mut h2, &version.to_le_bytes());

    CacheKey(splitmix64(h1), splitmix64(h2))
}

static LAST_SYNCED_RULES_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Check and synchronize cache with the active rules version.
/// If the version has changed since the last check, all cache shards are cleared to prevent stale entries.
pub fn sync_rules_version(current_version: u64) -> bool {
    let prev = LAST_SYNCED_RULES_VERSION.load(std::sync::atomic::Ordering::Acquire);
    if prev == current_version {
        return false;
    }
    let old = LAST_SYNCED_RULES_VERSION.swap(current_version, std::sync::atomic::Ordering::AcqRel);
    if old != 0 && old != current_version {
        clear();
        true
    } else {
        false
    }
}

#[inline]
fn splitmix64(mut hash: u64) -> u64 {
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^= hash >> 31;
    hash
}

/// Initialize the cache with the configured size.
///
/// The total capacity is split evenly across shards (each shard holds at
/// least one entry), so the aggregate bound matches the configured
/// `cache_size`; eviction order becomes per-shard LRU instead of global LRU,
/// which affects hit rate only — never translation output.
pub fn init() {
    let total = if config::is_initialized() {
        let cfg = config::get();
        if cfg.cache_size > 0 {
            cfg.cache_size
        } else {
            1
        }
    } else {
        1
    };
    let per_shard = total.div_ceil(SHARDS).max(1);
    for shard in &CACHE {
        *shard.0.write() = Some(LruCache::new(
            std::num::NonZeroUsize::new(per_shard).unwrap(),
        ));
    }
}

#[inline]
fn shard(key: CacheKey) -> &'static CacheShard {
    // Low bits of key.0 uniformly select the shard after splitmix64 diffusion.
    let index = (key.0 as usize) & (SHARDS - 1);
    &CACHE[index]
}

/// Read a value from the cache by key. Returns None if disabled or missing.
///
/// On cache hit, clones the `Arc<str>` reference directly without heap allocation
/// or string copying. Reads hold a shared lock, and optimistically update LRU order
/// only when the shard write lock is uncontended.
pub fn get(key: CacheKey) -> Option<std::sync::Arc<str>> {
    if !enabled() {
        return None;
    }
    let shard = shard(key);
    let hit = {
        let read_guard = shard.0.read();
        let cache = read_guard.as_ref()?;
        cache.peek(&key).filter(|v| !v.is_empty()).cloned()
    };

    let val = hit?;

    if let Some(mut write_guard) = shard.0.try_write() {
        if let Some(cache) = write_guard.as_mut() {
            cache.get(&key);
        }
    }

    Some(val)
}

/// Read a value from the cache by key. Returns None if disabled or missing.
///
/// On cache hit, clones the `Arc<str>` reference directly without heap allocation
/// or string copying. Alias for [`get`].
#[inline]
pub fn read_cached(key: CacheKey) -> Option<std::sync::Arc<str>> {
    get(key)
}

/// Write a value into the cache under a precomputed key.
pub fn set(key: CacheKey, result: impl Into<std::sync::Arc<str>>) {
    if !enabled() {
        return;
    }
    let val: std::sync::Arc<str> = result.into();
    let mut guard = shard(key).0.write();
    if let Some(cache) = guard.as_mut() {
        cache.put(key, val);
    }
}

/// Insert a value into the cache under a precomputed key. Alias for [`set`].
#[inline]
pub fn insert(key: CacheKey, result: impl Into<std::sync::Arc<str>>) {
    set(key, result);
}

/// Remove a specific entry from the cache by key. Returns true if the key was present.
pub fn invalidate(key: CacheKey) -> bool {
    if !enabled() {
        return false;
    }
    let mut guard = shard(key).0.write();
    if let Some(cache) = guard.as_mut() {
        cache.pop(&key).is_some()
    } else {
        false
    }
}

/// Write a value into the cache under a precomputed key.
#[inline]
pub fn write_cached(result: &str, key: CacheKey) {
    set(key, result);
}

/// Remove all cached entries across all shards.
pub fn clear() {
    for shard in &CACHE {
        let mut guard = shard.0.write();
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    }
}

/// Return the total number of cached entries across all shards.
pub fn len() -> usize {
    if !enabled() {
        return 0;
    }
    CACHE
        .iter()
        .map(|shard| shard.0.read().as_ref().map_or(0, |c| c.len()))
        .sum()
}

/// Return true if the cache contains no entries across all shards.
#[inline]
pub fn is_empty() -> bool {
    len() == 0
}

/// Check if the cache contains an entry for the given key without updating LRU order.
pub fn contains(key: CacheKey) -> bool {
    if !enabled() {
        return false;
    }
    let guard = shard(key).0.read();
    guard
        .as_ref()
        .is_some_and(|c| c.peek(&key).is_some_and(|v| !v.is_empty()))
}

/// Estimated memory usage in bytes of all cached entries across all shards,
/// including fixed shard structures.
pub fn memory_usage() -> usize {
    if !enabled() {
        return 0;
    }
    let mut total_bytes = std::mem::size_of_val(&CACHE);
    for shard in &CACHE {
        let guard = shard.0.read();
        if let Some(cache) = guard.as_ref() {
            // Per-entry LRU node overhead:
            // CacheKey (16 bytes) + Arc pointer (8 bytes) + hash table bucket (~16 bytes) + LRU links (~16 bytes)
            const ENTRY_OVERHEAD: usize = std::mem::size_of::<CacheKey>()
                + std::mem::size_of::<std::sync::Arc<str>>()
                + 32;

            for (_key, val) in cache.iter() {
                let arc_overhead = std::mem::size_of::<usize>() * 2;
                total_bytes += ENTRY_OVERHEAD + arc_overhead + val.len();
            }
        }
    }
    total_bytes
}

/// Translation cache handle and utility wrapper.
pub struct TranslationCache;

impl TranslationCache {
    #[inline]
    pub fn get(key: CacheKey) -> Option<std::sync::Arc<str>> {
        get(key)
    }

    #[inline]
    pub fn insert(key: CacheKey, result: impl Into<std::sync::Arc<str>>) {
        set(key, result);
    }

    #[inline]
    pub fn invalidate(key: CacheKey) -> bool {
        invalidate(key)
    }

    #[inline]
    pub fn clear() {
        clear();
    }

    #[inline]
    pub fn len() -> usize {
        len()
    }

    #[inline]
    pub fn is_empty() -> bool {
        is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use clap::Parser;
    use std::sync::Once;
    static INIT: Once = Once::new();
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup_test_cache() -> parking_lot::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock();
        INIT.call_once(|| {
            if !crate::config::is_initialized() {
                let cli =
                    crate::config::Cli::try_parse_from(["mtranserver", "--cache-size", "128"])
                        .unwrap();
                crate::config::init(cli);
            }
        });
        init();
        guard
    }

    #[test]
    fn test_cache_key_isomorphism_resistance() {
        let k1 = compute_key(&["a", "bc"]);
        let k2 = compute_key(&["ab", "c"]);
        assert_ne!(k1, k2);
        assert_ne!(k1.0, k2.0);
        assert_ne!(k1.1, k2.1);

        let k3 = compute_key(&["en", "zh", "hello"]);
        let k4 = compute_key(&["enzh", "", "hello"]);
        assert_ne!(k3, k4);

        let k5 = compute_key(&["en", "zh", "hello"]);
        let k6 = compute_key(&["en", "zhhello", ""]);
        assert_ne!(k5, k6);

        let k_empty_1 = compute_key(&["", "abc"]);
        let k_empty_2 = compute_key(&["abc", ""]);
        assert_ne!(k_empty_1, k_empty_2);

        let k_same_a = compute_key(&["en", "zh", "deterministic"]);
        let k_same_b = compute_key(&["en", "zh", "deterministic"]);
        assert_eq!(k_same_a, k_same_b);
    }

    #[test]
    fn test_cache_key_with_version_and_sync() {
        let _guard = setup_test_cache();
        LAST_SYNCED_RULES_VERSION.store(0, std::sync::atomic::Ordering::Release);
        let k_v1 = compute_key_with_version(&["en", "zh", "test text"], 1);
        let k_v2 = compute_key_with_version(&["en", "zh", "test text"], 2);
        assert_ne!(k_v1, k_v2);

        let k_v1_again = compute_key_with_version(&["en", "zh", "test text"], 1);
        assert_eq!(k_v1, k_v1_again);

        // Test version sync triggers cache eviction
        write_cached("result_v1", k_v1);
        assert!(contains(k_v1));
        assert_eq!(read_cached(k_v1).as_deref(), Some("result_v1"));

        // Syncing with same version does not flush
        assert!(!sync_rules_version(1));
        assert!(contains(k_v1));

        // Advancing version triggers cache clear
        assert!(sync_rules_version(2));
        assert!(!contains(k_v1));
    }

    #[test]
    fn test_cache_key_shard_distribution() {
        let mut shard_hits = [false; SHARDS];
        for i in 0..100 {
            let key = compute_key(&["en", "zh", &format!("text_{}", i)]);
            let idx = (key.0 as usize) & (SHARDS - 1);
            shard_hits[idx] = true;
        }
        let hit_count = shard_hits.iter().filter(|&&h| h).count();
        assert!(
            hit_count >= 10,
            "Expected broad shard distribution, got {hit_count}/{SHARDS}"
        );
    }

    #[test]
    fn test_cache_shard_cacheline_alignment() {
        // Each shard MUST be 64-byte aligned to prevent L1D false sharing between CPU cores.
        assert_eq!(std::mem::align_of::<CacheShard>(), 64);
        assert!(std::mem::size_of::<CacheShard>() >= 64);
        assert_eq!(std::mem::size_of::<CacheShard>() % 64, 0);

        // Static array elements must each reside on distinct 64-byte aligned boundaries.
        let base_ptr = std::ptr::addr_of!(CACHE[0]) as usize;
        assert_eq!(base_ptr % 64, 0, "Base shard pointer is not 64-byte aligned");
        for i in 1..SHARDS {
            let ptr = std::ptr::addr_of!(CACHE[i]) as usize;
            assert_eq!(ptr % 64, 0, "Shard {i} pointer is not 64-byte aligned");
            assert!(
                ptr - base_ptr >= i * 64,
                "Shard {i} overlaps with preceding cache lines"
            );
        }
    }

    #[test]
    fn test_cache_operations() {
        let _lock = setup_test_cache();
        assert!(enabled());
        assert!(is_empty());

        let k1 = compute_key(&["en", "zh", "test_op_1"]);
        let k2 = compute_key(&["en", "zh", "test_op_2"]);

        assert!(!contains(k1));
        assert_eq!(get(k1), None);

        set(k1, "value_1");
        assert!(contains(k1));
        assert_eq!(get(k1).as_deref(), Some("value_1"));
        assert_eq!(read_cached(k1).as_deref(), Some("value_1"));
        assert_eq!(len(), 1);
        assert!(!is_empty());

        // Set via Arc<str>
        let arc_val: std::sync::Arc<str> = std::sync::Arc::from("value_2");
        set(k2, arc_val);
        assert!(contains(k2));
        assert_eq!(get(k2).as_deref(), Some("value_2"));
        assert_eq!(len(), 2);

        let mem = memory_usage();
        assert!(mem > std::mem::size_of_val(&CACHE));

        clear();
        assert_eq!(len(), 0);
        assert!(is_empty());
        assert!(!contains(k1));
        assert!(!contains(k2));
    }

    #[test]
    fn test_cache_read_write_zero_copy() {
        let _lock = setup_test_cache();
        assert!(enabled());
        let key = compute_key(&["en", "zh", "zero copy test string"]);
        write_cached("零拷贝测试结果", key);

        let cached1 = read_cached(key).expect("cache should hit");
        assert_eq!(&*cached1, "零拷贝测试结果");
        assert_eq!(std::sync::Arc::strong_count(&cached1), 2);

        let cached2 = read_cached(key).expect("cache should hit again");
        assert_eq!(&*cached2, "零拷贝测试结果");
        assert_eq!(std::sync::Arc::strong_count(&cached2), 3);
        assert!(std::sync::Arc::ptr_eq(&cached1, &cached2));

        let missing_key = compute_key(&["en", "zh", "nonexistent"]);
        assert!(read_cached(missing_key).is_none());
    }
}
