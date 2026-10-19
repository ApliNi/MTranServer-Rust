// Shared authentication primitives: constant-time token comparison and the
// unified credential extractor used by the auth middleware and by the plugin
// endpoints that perform their own auth check.

use axum::http::HeaderMap;
use sha2::{Digest, Sha256};
use std::borrow::Cow;

/// Constant-time comparison between a candidate token and a precomputed SHA-256 digest:
/// the candidate token is hashed with SHA-256, and folded XOR runs over the 32-byte digests.
pub fn token_matches_hash(token: &str, expected_hash: &[u8; 32]) -> bool {
    let dt = Sha256::digest(token.as_bytes());
    let mut diff = 0u8;
    for (x, y) in dt.iter().zip(expected_hash.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Constant-time token comparison: both sides are hashed with SHA-256 first,
/// so the folded XOR below runs over fixed-size digests and timing leaks
/// neither length nor prefix information about the configured token.
pub fn tokens_equal(a: &str, b: &str) -> bool {
    let db: [u8; 32] = Sha256::digest(b.as_bytes()).into();
    token_matches_hash(a, &db)
}

struct CachedDigest {
    token: String,
    hash: [u8; 32],
}

static EXPECTED_DIGEST: parking_lot::RwLock<Option<CachedDigest>> = parking_lot::RwLock::new(None);

/// Retrieve or compute the SHA-256 digest of the expected API token.
pub fn get_or_compute_expected_digest(expected: &str) -> [u8; 32] {
    {
        let read = EXPECTED_DIGEST.read();
        if let Some(cached) = &*read {
            if cached.token == expected {
                return cached.hash;
            }
        }
    }

    let mut write = EXPECTED_DIGEST.write();
    if let Some(cached) = &*write {
        if cached.token == expected {
            return cached.hash;
        }
    }

    let hash: [u8; 32] = Sha256::digest(expected.as_bytes()).into();
    *write = Some(CachedDigest {
        token: expected.to_string(),
        hash,
    });
    hash
}

/// Minimal percent-decoding for query parameter values: `%XX` escapes are
/// decoded, malformed escapes are kept verbatim. `+` is intentionally NOT
/// treated as a space — generated tokens are alphanumeric/hex and never
/// contain spaces, so a literal `+` survives uncorrupted. (No
/// `form_urlencoded` dependency is needed for this.)
fn percent_decode(value: &str) -> Cow<'_, str> {
    if !value.contains('%') {
        return Cow::Borrowed(value);
    }
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && bytes[i + 1].is_ascii_hexdigit()
            && bytes[i + 2].is_ascii_hexdigit()
        {
            let hi = (bytes[i + 1] as char).to_digit(16).unwrap() as u8;
            let lo = (bytes[i + 2] as char).to_digit(16).unwrap() as u8;
            out.push(hi << 4 | lo);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Cow::Owned(String::from_utf8_lossy(&out).into_owned())
}

/// Extract the API credential from a request, matching the TS
/// `expressAuthentication` priority order: Authorization header (Bearer and
/// DeepL-Auth-Key prefixes stripped) → query `api_token` → query `token` →
/// `x-api-token` header. `query` is the raw query string (handlers:
/// `RawQuery`), so values are percent-decoded here. Returns `None` when no
/// credential is present.
pub fn extract_auth_token<'a>(
    headers: &'a HeaderMap,
    query: Option<&'a str>,
) -> Option<Cow<'a, str>> {
    // 1. Authorization header (strip "Bearer " like TS `replace('Bearer ', '')`;
    // DeepL clients send "DeepL-Auth-Key <token>", stripped the same way)
    if let Some(val) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        let trimmed = val.trim();
        if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
            return Some(Cow::Borrowed(trimmed[7..].trim()));
        }
        if trimmed.len() >= 15 && trimmed[..15].eq_ignore_ascii_case("deepl-auth-key ") {
            return Some(Cow::Borrowed(trimmed[15..].trim()));
        }
    }

    // 2. query `api_token`, then query `token`
    if let Some(query) = query {
        for name in ["api_token", "token"] {
            for pair in query.split('&') {
                let mut parts = pair.splitn(2, '=');
                if parts.next() == Some(name) {
                    if let Some(raw) = parts.next() {
                        return Some(percent_decode(raw));
                    }
                }
            }
        }
    }

    // 3. x-api-token header
    headers
        .get("x-api-token")
        .and_then(|v| v.to_str().ok())
        .map(Cow::Borrowed)
}

/// Verify request credentials against the configured API token using
/// constant-time comparison. Callers gate on `config.api_token` being set.
pub fn verify_request(headers: &HeaderMap, query: Option<&str>, expected: &str) -> bool {
    match extract_auth_token(headers, query) {
        Some(token) => {
            let expected_hash = get_or_compute_expected_digest(expected);
            token_matches_hash(&token, &expected_hash)
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn test_token_matches_hash_and_tokens_equal() {
        let token = "mock_test_token_123";
        let hash: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        assert!(token_matches_hash(token, &hash));
        assert!(!token_matches_hash("wrong_token", &hash));

        assert!(tokens_equal(token, token));
        assert!(!tokens_equal(token, "wrong_token"));
    }

    #[test]
    fn test_extract_auth_token_borrowed_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer mock_test_token_abc"),
        );
        let extracted = extract_auth_token(&headers, None);
        assert!(matches!(extracted, Some(Cow::Borrowed("mock_test_token_abc"))));

        headers.insert(
            "authorization",
            HeaderValue::from_static("DeepL-Auth-Key mock_test_deepl_key"),
        );
        let extracted = extract_auth_token(&headers, None);
        assert!(matches!(extracted, Some(Cow::Borrowed("mock_test_deepl_key"))));

        headers.remove("authorization");
        headers.insert("x-api-token", HeaderValue::from_static("header_tok"));
        let extracted = extract_auth_token(&headers, None);
        assert!(matches!(extracted, Some(Cow::Borrowed("header_tok"))));

        headers.insert(
            "authorization",
            HeaderValue::from_static("bearer lowercase_token"),
        );
        let extracted = extract_auth_token(&headers, None);
        assert!(matches!(extracted, Some(Cow::Borrowed("lowercase_token"))));

        headers.insert(
            "authorization",
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        let extracted_fallback = extract_auth_token(&headers, Some("token=fallback_query_tok"));
        assert!(matches!(extracted_fallback, Some(Cow::Borrowed("fallback_query_tok"))));
    }
    #[test]
    fn test_extract_auth_token_query() {
        let headers = HeaderMap::new();
        let extracted = extract_auth_token(&headers, Some("api_token=query_tok&other=1"));
        assert!(matches!(extracted, Some(Cow::Borrowed("query_tok"))));

        let extracted = extract_auth_token(&headers, Some("token=hello%20world"));
        assert_eq!(extracted.as_deref(), Some("hello world"));
        assert!(matches!(extracted, Some(Cow::Owned(_))));
    }

    #[test]
    fn test_verify_request() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer mock_test_token"));
        assert!(verify_request(&headers, None, "mock_test_token"));
        assert!(!verify_request(&headers, None, "other_token"));

        let empty_headers = HeaderMap::new();
        assert!(!verify_request(&empty_headers, None, "mock_test_token"));
    }

    #[test]
    fn test_get_or_compute_expected_digest() {
        let h1 = get_or_compute_expected_digest("token_abc");
        let expected: [u8; 32] = Sha256::digest(b"token_abc").into();
        assert_eq!(h1, expected);

        // Second read from fast path
        let h2 = get_or_compute_expected_digest("token_abc");
        assert_eq!(h2, expected);

        // Recomputation on new token
        let h3 = get_or_compute_expected_digest("token_xyz");
        let expected_xyz: [u8; 32] = Sha256::digest(b"token_xyz").into();
        assert_eq!(h3, expected_xyz);
    }
}
