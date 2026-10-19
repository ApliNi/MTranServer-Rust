pub mod arch;
pub mod auth;
pub mod cache;
pub mod lang;
pub mod profiling;

/// Compare two version strings (e.g. "1.1.0" vs "1.0.0", "v1.2" vs "1.1.9").
/// Returns std::cmp::Ordering: Less if v1 < v2, Equal if v1 == v2, Greater if v1 > v2.
#[inline]
pub fn compare_versions(v1: &str, v2: &str) -> std::cmp::Ordering {
    let parse_components = |s: &str| -> Vec<u64> {
        let clean = s.trim().trim_start_matches(|c: char| c == 'v' || c == 'V');
        clean
            .split(['.', '-', '_'])
            .filter_map(|p| p.parse::<u64>().ok())
            .collect()
    };

    let c1 = parse_components(v1);
    let c2 = parse_components(v2);
    if !c1.is_empty() || !c2.is_empty() {
        let max_len = c1.len().max(c2.len());
        for i in 0..max_len {
            let a = c1.get(i).copied().unwrap_or(0);
            let b = c2.get(i).copied().unwrap_or(0);
            match a.cmp(&b) {
                std::cmp::Ordering::Equal => continue,
                non_eq => return non_eq,
            }
        }
        std::cmp::Ordering::Equal
    } else {
        v1.trim().cmp(v2.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_versions() {
        assert_eq!(compare_versions("v2.1", "v1.9"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("1.0.0", "1.0"), std::cmp::Ordering::Equal);
        assert_eq!(compare_versions("v1.2.0", "1.2.0"), std::cmp::Ordering::Equal);
        assert_eq!(compare_versions("1.10.0", "1.2.0"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("0.9.1", "1.0.0"), std::cmp::Ordering::Less);
    }
}
