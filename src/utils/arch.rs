//! Runtime CPU feature detection used by the compute kernels.

use std::sync::LazyLock;

/// The highest SIMD instruction set available on the current CPU.
///
/// Each tier is a strict superset of the previous ones, so a kernel is chosen
/// as the highest level whose feature flags are all present. Order matters:
/// the enum variants are ordered from weakest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// Portable scalar fallback (also used on non-x86/ARM targets).
    Scalar,
    /// 128-bit SSE gain without x86-64 SIMD acceleration of interest.
    Sse41,
    /// 256-bit AVX (no FMA).
    Avx,
    /// 256-bit AVX2 + FMA (best on AMD/Intel x86-64).
    Avx2Fma,
    /// 128-bit ARM NEON (aarch64). Never constructed on x86 builds.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Neon,
}

/// Highest SIMD level available, evaluated once and memoised.
///
/// `MT_FORCE_SIMD` overrides detection (`scalar`, `sse41`, `avx`, `avx2`); an unknown
/// value is ignored. This exists so the lower kernel tiers can be exercised
/// on newer CPUs — feature detection alone only ever reaches the top tier.
pub fn simd_level() -> SimdLevel {
    static LEVEL: LazyLock<SimdLevel> = LazyLock::new(|| {
        resolve_simd_level(std::env::var("MT_FORCE_SIMD").as_deref().ok())
    });
    *LEVEL
}

pub(crate) fn parse_simd_override(s: &str) -> Option<SimdLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "scalar" | "none" => Some(SimdLevel::Scalar),
        "sse41" => Some(SimdLevel::Sse41),
        "avx" => Some(SimdLevel::Avx),
        "avx2" => Some(SimdLevel::Avx2Fma),
        #[cfg(target_arch = "aarch64")]
        "neon" => Some(SimdLevel::Neon),
        _ => None,
    }
}

pub(crate) fn resolve_simd_level(forced: Option<&str>) -> SimdLevel {
    forced
        .and_then(parse_simd_override)
        .unwrap_or_else(detect_simd_level)
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn forced_level_disables_avx2(forced: Option<&str>) -> bool {
    forced
        .and_then(parse_simd_override)
        .is_some_and(|lvl| matches!(lvl, SimdLevel::Scalar | SimdLevel::Sse41 | SimdLevel::Avx))
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn forced_simd_disables_avx2() -> bool {
    forced_level_disables_avx2(std::env::var("MT_FORCE_SIMD").as_deref().ok())
}

/// Returns whether the host CPU supports AVX2.
/// Evaluated once and cached. Respects `MT_FORCE_SIMD`.
pub fn has_avx2() -> bool {
    static CACHED: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if forced_simd_disables_avx2() {
                return false;
            }
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    });
    *CACHED
}

/// Returns whether the host CPU supports AVX-VNNI (VEX-encoded) or AVX-512 VNNI (EVEX-encoded).
///
/// Both instruction sets provide `vpdpbusd` for accelerating 8-bit quantized matrix
/// multiplication: AVX-VNNI uses VEX encoding (Alder Lake+, Zen 5), while AVX-512 VNNI
/// uses EVEX encoding (Cascade Lake+, Ice Lake+, Zen 4+).
/// Evaluated once and cached. Respects `MT_FORCE_SIMD`.
pub fn has_avxvnni() -> bool {
    static CACHED: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if forced_simd_disables_avx2() {
                return false;
            }
            std::arch::is_x86_feature_detected!("avxvnni")
                || (std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx512vl"))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    });
    *CACHED
}

/// Returns whether the host CPU supports EVEX-encoded AVX-512 VNNI (`vpdpbusd` with 256-bit VL).
///
/// Available on Intel Cascade Lake/Ice Lake/Sapphire Rapids and AMD Zen 4+.
/// Evaluated once and cached. Respects `MT_FORCE_SIMD`.
pub fn has_avx512vnni() -> bool {
    static CACHED: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if forced_simd_disables_avx2() {
                return false;
            }
            std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    });
    *CACHED
}

/// Returns whether VNNI acceleration should use EVEX encoding (AVX-512 VNNI) rather than VEX (AVX-VNNI).
///
/// Returns `true` when the CPU supports AVX-512 VNNI (`avx512vnni` + `avx512vl`) but does NOT support
/// VEX-encoded AVX-VNNI (e.g. Intel Cascade Lake, Ice Lake, Cooper Lake, AMD Zen 4).
pub fn vnni_uses_evex() -> bool {
    static CACHED: LazyLock<bool> = LazyLock::new(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if forced_simd_disables_avx2() {
                return false;
            }
            !std::arch::is_x86_feature_detected!("avxvnni")
                && (std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx512vl"))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    });
    *CACHED
}

fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64.
        return SimdLevel::Neon;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return SimdLevel::Avx2Fma;
        }
        if std::arch::is_x86_feature_detected!("avx") {
            return SimdLevel::Avx;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return SimdLevel::Sse41;
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return SimdLevel::Scalar;
    }
    #[allow(unreachable_code)]
    SimdLevel::Scalar
}

/// Human-readable report of the SIMD acceleration techniques this build can
/// use and whether each is actually active on the current CPU.
///
/// Each entry is `name: <bool>` — `true` when the build ships a kernel for
/// that instruction set *and* the running CPU supports it, `false` when only
/// one of those holds. The trailing `active:` line names the tier that the
/// compute kernels dispatch to (see [`simd_level`]).
pub fn acceleration_report() -> String {
    let mut parts: Vec<String> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
        let avx = std::arch::is_x86_feature_detected!("avx");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let fma = std::arch::is_x86_feature_detected!("fma");
        let vnni = has_avxvnni();
        let avx512vnni = has_avx512vnni();
        parts.push(format!("sse4.1: {sse41}"));
        parts.push(format!("avx: {avx}"));
        parts.push(format!("avx2: {avx2}"));
        parts.push(format!("fma: {fma}"));
        parts.push(format!("vnni: {vnni}"));
        parts.push(format!("avx512vnni: {avx512vnni}"));
        let active = match simd_level() {
            SimdLevel::Avx2Fma => "avx2+fma",
            SimdLevel::Avx => "avx",
            SimdLevel::Sse41 => "sse4.1",
            SimdLevel::Scalar | SimdLevel::Neon => "scalar",
        };
        parts.push(format!("active: {active}"));
    }
    #[cfg(target_arch = "aarch64")]
    {
        parts.push("neon: true".to_string());
        let active = match simd_level() {
            SimdLevel::Neon => "neon",
            _ => "scalar",
        };
        parts.push(format!("active: {active}"));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        parts.push("active: scalar".to_string());
    }
    parts.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simd_override() {
        assert_eq!(parse_simd_override("scalar"), Some(SimdLevel::Scalar));
        assert_eq!(parse_simd_override("none"), Some(SimdLevel::Scalar));
        assert_eq!(parse_simd_override("Scalar"), Some(SimdLevel::Scalar));
        assert_eq!(parse_simd_override("  NONE  "), Some(SimdLevel::Scalar));
        assert_eq!(parse_simd_override("sse41"), Some(SimdLevel::Sse41));
        assert_eq!(parse_simd_override("avx"), Some(SimdLevel::Avx));
        assert_eq!(parse_simd_override("avx2"), Some(SimdLevel::Avx2Fma));
        assert_eq!(parse_simd_override("unknown"), None);
        assert_eq!(parse_simd_override(""), None);
    }

    #[test]
    fn test_resolve_simd_level_overrides() {
        assert_eq!(resolve_simd_level(Some("scalar")), SimdLevel::Scalar);
        assert_eq!(resolve_simd_level(Some("none")), SimdLevel::Scalar);
        assert_eq!(resolve_simd_level(Some("sse41")), SimdLevel::Sse41);
        assert_eq!(resolve_simd_level(Some("avx")), SimdLevel::Avx);
        assert_eq!(resolve_simd_level(Some("avx2")), SimdLevel::Avx2Fma);
        assert_eq!(resolve_simd_level(Some("invalid")), detect_simd_level());
        assert_eq!(resolve_simd_level(None), detect_simd_level());
    }

    #[test]
    fn test_forced_disables_avx2() {
        assert!(forced_level_disables_avx2(Some("scalar")));
        assert!(forced_level_disables_avx2(Some("none")));
        assert!(forced_level_disables_avx2(Some("sse41")));
        assert!(forced_level_disables_avx2(Some("avx")));
        assert!(!forced_level_disables_avx2(Some("avx2")));
        assert!(!forced_level_disables_avx2(Some("unknown")));
        assert!(!forced_level_disables_avx2(None));
    }

    #[test]
    fn test_force_simd_scalar_env_subprocess() {
        let exe = std::env::current_exe().expect("current exe");
        let output = std::process::Command::new(exe)
            .env("MT_FORCE_SIMD", "scalar")
            .args(["--nocapture", "utils::arch::tests::helper_verify_scalar_env", "--exact"])
            .output()
            .expect("spawn helper");
        assert!(
            output.status.success(),
            "subprocess failed: stdout={}, stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn helper_verify_scalar_env() {
        if std::env::var("MT_FORCE_SIMD").as_deref() != Ok("scalar") {
            return;
        }
        assert_eq!(simd_level(), SimdLevel::Scalar);
        assert!(!has_avx2());
        assert!(!has_avxvnni());
        assert!(!has_avx512vnni());
        assert!(!vnni_uses_evex());
        assert!(acceleration_report().contains("active: scalar"));
    }

    #[test]
    fn test_force_simd_none_env_subprocess() {
        let exe = std::env::current_exe().expect("current exe");
        let output = std::process::Command::new(exe)
            .env("MT_FORCE_SIMD", "none")
            .args(["--nocapture", "utils::arch::tests::helper_verify_none_env", "--exact"])
            .output()
            .expect("spawn helper");
        assert!(
            output.status.success(),
            "subprocess failed: stdout={}, stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn helper_verify_none_env() {
        if std::env::var("MT_FORCE_SIMD").as_deref() != Ok("none") {
            return;
        }
        assert_eq!(simd_level(), SimdLevel::Scalar);
        assert!(!has_avx2());
        assert!(!has_avxvnni());
        assert!(!has_avx512vnni());
        assert!(!vnni_uses_evex());
        assert!(acceleration_report().contains("active: scalar"));
    }
}
