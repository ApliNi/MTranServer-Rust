// Request/response middleware: auth, logging, error formatting.

use axum::{
    body::Body, body::HttpBody, extract::Request, http::StatusCode, middleware::Next,
    response::Response,
};
use serde::{Deserialize, Serialize};

pub(crate) const UNAUTHORIZED_BODY: &str = r#"{"error":"Unauthorized"}"#;
pub(crate) const TOO_MANY_REQUESTS_BODY: &str =
    r#"{"error":"Too Many Requests","message":"Too Many Requests"}"#;

#[derive(Deserialize)]
struct IncomingError<'a> {
    #[serde(borrow)]
    error: Option<&'a str>,
    #[serde(borrow)]
    message: Option<&'a str>,
}

#[derive(Serialize)]
struct OutgoingError<'a> {
    error: &'a str,
    message: &'a str,
}

use crate::config;

/// Auth middleware: checks api_token from config against request credentials.
pub async fn auth_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    let cfg = config::get();
    if cfg.api_token.is_empty() {
        return Ok(next.run(req).await);
    }

    // The raw query string is needed so the shared extractor can decode it.
    let query = req.uri().query();

    if crate::utils::auth::verify_request(req.headers(), query, &cfg.api_token) {
        return Ok(next.run(req).await);
    }

    // Return 401 with the exact TS error shape.
    Ok(Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header("content-type", "application/json")
        .body(Body::from(UNAUTHORIZED_BODY))
        .unwrap())
}

/// Error response formatter. Handlers return faithful status codes via
/// `ApiError`; this middleware only decorates the response:
/// - 401 → `{ error: 'Unauthorized' }`
/// - other errors → `{ error, message }` with the handler's status
/// - unmatched routes keep TS's plain-text `404`.
pub async fn error_response_formatter(req: Request, next: Next) -> Result<Response, StatusCode> {
    let response = next.run(req).await;

    let status = response.status();
    if status.is_success() || status.is_redirection() {
        return Ok(response);
    }

    let (mut parts, body) = response.into_parts();

    if status == StatusCode::NOT_FOUND {
        parts.headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/plain; charset=utf-8"),
        );
        return Ok(Response::from_parts(parts, Body::from("404")));
    }

    if status == StatusCode::UNAUTHORIZED {
        parts.headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        return Ok(Response::from_parts(parts, Body::from(UNAUTHORIZED_BODY)));
    }

    if status == StatusCode::TOO_MANY_REQUESTS {
        parts.headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        return Ok(Response::from_parts(parts, Body::from(TOO_MANY_REQUESTS_BODY)));
    }

    if let Some(ct) = parts.headers.get(axum::http::header::CONTENT_TYPE) {
        if let Ok(ct_str) = ct.to_str() {
            if !ct_str.contains("json") {
                return Ok(Response::from_parts(parts, body));
            }
        }
    }

    let body_bytes = axum::body::to_bytes(body, 64 * 1024)
        .await
        .unwrap_or_default();

    let Ok(incoming) = serde_json::from_slice::<IncomingError>(&body_bytes) else {
        return Ok(Response::from_parts(parts, Body::from(body_bytes)));
    };

    if incoming.error.is_some() && incoming.message.is_some() {
        return Ok(Response::from_parts(parts, Body::from(body_bytes)));
    }

    let error = incoming.error.unwrap_or("Internal Server Error");
    let message = incoming.message.unwrap_or(error);
    let outgoing = OutgoingError { error, message };
    let body_str = serde_json::to_string(&outgoing).unwrap_or_default();
    parts.headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    Ok(Response::from_parts(parts, Body::from(body_str)))
}

/// CORS middleware replicating the TS `cors.ts`: static allow-list headers,
/// and OPTIONS preflight short-circuits with a 204.
pub async fn cors_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    if req.method() == axum::http::Method::OPTIONS {
        let mut resp = Response::new(Body::empty());
        *resp.status_mut() = StatusCode::NO_CONTENT;
        set_cors_headers(resp.headers_mut());
        return Ok(resp);
    }

    let mut resp = next.run(req).await;
    set_cors_headers(resp.headers_mut());
    Ok(resp)
}

fn set_cors_headers(headers: &mut axum::http::HeaderMap) {
    use axum::http::header::{HeaderName, HeaderValue};
    macro_rules! set {
        ($name:literal, $value:literal) => {
            headers.insert(
                HeaderName::from_static($name),
                HeaderValue::from_static($value),
            );
        };
    }
    set!("access-control-allow-origin", "*");
    set!(
        "access-control-allow-methods",
        "GET, POST, PUT, DELETE, OPTIONS"
    );
    set!(
        "access-control-allow-headers",
        "Content-Type, Authorization, X-API-Token"
    );
    set!("access-control-max-age", "86400");
}

/// Request logging middleware. Only active when config.log_requests is true.
/// Format mirrors TS `request-logger.ts`:
/// `${method} ${status} ${path} ${duration}s ${bytes}b - ${ip}`
pub async fn log_requests(req: Request, next: Next) -> Result<Response, StatusCode> {
    let cfg = config::get();
    if !cfg.log_requests {
        return Ok(next.run(req).await);
    }

    let method = req.method().clone();
    // Use the actual request path (TS logs `req.path`), not the axum route
    // pattern: `MatchedPath` would render captures as `/ui/{*path}` instead
    // of the real `/ui/app.js` or `/style.css`.
    let path = req.uri().path().to_string();

    let ip = extract_client_ip(&req);
    let started = std::time::Instant::now();
    let response = next.run(req).await;
    let elapsed = started.elapsed();

    // TS: `((Date.now() - start) / 1000).toFixed(2)` → seconds, 2 decimals.
    let duration = format!("{:.2}", elapsed.as_secs_f64());
    // TS: `res.get('content-length') || 0`. axum fills content-length at
    // send time, so fall back to the body's size hint when the header is
    // absent in middleware.
    let bytes = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            response
                .body()
                .size_hint()
                .exact()
                .map(|n| n.to_string())
                .unwrap_or_else(|| "0".to_string())
        });

    crate::logger::important(&format!(
        "{} {} {} {}s {}b - {}",
        method,
        response.status().as_u16(),
        path,
        duration,
        bytes,
        ip
    ));

    Ok(response)
}

/// Concurrency limit middleware: bounds concurrent in-flight HTTP requests via a shared Semaphore.
/// When maximum concurrency is reached, immediately rejects incoming requests with 429 Too Many Requests.
pub async fn concurrency_limit_middleware(
    axum::extract::State(semaphore): axum::extract::State<std::sync::Arc<tokio::sync::Semaphore>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let _permit = match semaphore.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            return Ok(Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header("content-type", "application/json")
                .body(Body::from(TOO_MANY_REQUESTS_BODY))
                .unwrap());
        }
    };
    Ok(next.run(req).await)
}

/// Extract client IP from connection info or headers (x-real-ip, x-forwarded-for).
/// When `--trust-proxy` is set, headers take priority. Otherwise, direct socket address
/// is prioritized to prevent header spoofing, falling back to headers only if no socket info exists.
pub fn extract_client_ip(req: &Request) -> String {
    let trust_proxy = crate::config::is_initialized() && crate::config::get().trust_proxy;

    let extract_header = || {
        req.headers()
            .get("x-real-ip")
            .or_else(|| req.headers().get("x-forwarded-for"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(',').next().unwrap_or(s).trim().to_string())
            .filter(|s| !s.is_empty())
    };

    let extract_socket = || {
        req.extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip().to_string())
    };

    if trust_proxy {
        extract_header()
            .or_else(extract_socket)
            .unwrap_or_else(|| "unknown".to_string())
    } else {
        extract_socket()
            .or_else(extract_header)
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// Per-IP concurrency limiter tracking active in-flight requests per client IP.
pub struct IpConcurrencyLimiter {
    max_per_ip: usize,
    shards: [parking_lot::Mutex<std::collections::HashMap<std::sync::Arc<str>, usize>>; 16],
}

/// RAII Guard that decrements and cleans up per-IP concurrency count on drop.
pub struct IpConcurrencyGuard {
    limiter: std::sync::Arc<IpConcurrencyLimiter>,
    ip: std::sync::Arc<str>,
}

impl IpConcurrencyLimiter {
    pub const SHARDS: usize = 16;

    pub fn new(max_per_ip: usize) -> Self {
        Self {
            max_per_ip,
            shards: std::array::from_fn(|_| {
                parking_lot::Mutex::new(std::collections::HashMap::new())
            }),
        }
    }

    fn shard(&self, ip: &str) -> &parking_lot::Mutex<std::collections::HashMap<std::sync::Arc<str>, usize>> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ip.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % Self::SHARDS;
        &self.shards[idx]
    }

    pub fn try_acquire(self: &std::sync::Arc<Self>, ip: &str) -> Option<IpConcurrencyGuard> {
        let mut map = self.shard(ip).lock();
        let count = map.get(ip).copied().unwrap_or(0);
        if count >= self.max_per_ip {
            None
        } else {
            let ip_arc = match map.get_mut(ip) {
                Some(cnt) => {
                    *cnt += 1;
                    std::sync::Arc::from(ip)
                }
                None => {
                    let arc: std::sync::Arc<str> = ip.into();
                    map.insert(arc.clone(), 1);
                    arc
                }
            };
            Some(IpConcurrencyGuard {
                limiter: self.clone(),
                ip: ip_arc,
            })
        }
    }
}

impl Drop for IpConcurrencyGuard {
    fn drop(&mut self) {
        let mut map = self.limiter.shard(&self.ip).lock();
        if let Some(cnt) = map.get_mut(&*self.ip) {
            *cnt -= 1;
            if *cnt == 0 {
                map.remove(&*self.ip);
            }
        }
    }
}

/// Per-IP concurrency limit middleware: bounds concurrent in-flight HTTP requests per client IP.
/// When the client IP exceeds max concurrency, immediately rejects with 429 Too Many Requests.
pub async fn ip_concurrency_limit_middleware(
    axum::extract::State(limiter): axum::extract::State<std::sync::Arc<IpConcurrencyLimiter>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = extract_client_ip(&req);
    let _guard = match limiter.try_acquire(&ip) {
        Some(guard) => guard,
        None => {
            return Ok(Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header("content-type", "application/json")
                .body(Body::from(TOO_MANY_REQUESTS_BODY))
                .unwrap());
        }
    };
    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_client_ip_headers_and_fallback() {
        let req1 = Request::builder()
            .header("x-real-ip", "192.168.1.100")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req1), "192.168.1.100");

        let req2 = Request::builder()
            .header("x-forwarded-for", "10.0.0.1, 10.0.0.2")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_client_ip(&req2), "10.0.0.1");

        let req3 = Request::builder().body(Body::empty()).unwrap();
        assert_eq!(extract_client_ip(&req3), "unknown");
    }

    #[test]
    fn test_ip_concurrency_limiter_acquire_and_drop() {
        let limiter = std::sync::Arc::new(IpConcurrencyLimiter::new(1));
        let guard1 = limiter.try_acquire("127.0.0.1");
        assert!(guard1.is_some());

        let guard2 = limiter.try_acquire("127.0.0.1");
        assert!(guard2.is_none());

        // Other IP is allowed
        let guard_other = limiter.try_acquire("127.0.0.2");
        assert!(guard_other.is_some());

        drop(guard1);

        // After drop, 127.0.0.1 should be acquirable again
        let guard3 = limiter.try_acquire("127.0.0.1");
        assert!(guard3.is_some());
    }

    #[test]
    fn test_ip_concurrency_limiter_multiple_ips_and_shards() {
        let limiter = std::sync::Arc::new(IpConcurrencyLimiter::new(2));
        let mut guards = Vec::new();
        for i in 0..32 {
            let ip = format!("192.168.1.{}", i);
            let g1 = limiter.try_acquire(&ip);
            assert!(g1.is_some());
            let g2 = limiter.try_acquire(&ip);
            assert!(g2.is_some());
            let g3 = limiter.try_acquire(&ip);
            assert!(g3.is_none());
            guards.push(g1);
            guards.push(g2);
        }
        // Dropping all guards cleans up correctly
        guards.clear();
        for i in 0..32 {
            let ip = format!("192.168.1.{}", i);
            assert!(limiter.try_acquire(&ip).is_some());
        }
    }

    #[tokio::test]
    async fn test_error_response_formatter_unauthorized() {
        use axum::middleware;
        use tower::ServiceExt;

        let app = axum::Router::new()
            .route(
                "/unauth",
                axum::routing::get(|| async { StatusCode::UNAUTHORIZED }),
            )
            .layer(middleware::from_fn(error_response_formatter));

        let req = Request::builder().uri("/unauth").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(bytes, UNAUTHORIZED_BODY);
    }

    #[tokio::test]
    async fn test_error_response_formatter_not_found() {
        use axum::middleware;
        use tower::ServiceExt;

        let app = axum::Router::new().layer(middleware::from_fn(error_response_formatter));

        let req = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/plain; charset=utf-8"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(bytes, "404");
    }

    #[tokio::test]
    async fn test_error_response_formatter_decorates_error_json() {
        use axum::middleware;
        use tower::ServiceExt;

        let app = axum::Router::new()
            .route(
                "/err",
                axum::routing::get(|| async {
                    (
                        StatusCode::BAD_REQUEST,
                        axum::Json(serde_json::json!({ "error": "Bad Request" })),
                    )
                }),
            )
            .layer(middleware::from_fn(error_response_formatter));

        let req = Request::builder().uri("/err").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"], "Bad Request");
        assert_eq!(json["message"], "Bad Request");
    }

    #[tokio::test]
    async fn test_error_response_formatter_passes_non_json() {
        use axum::middleware;
        use tower::ServiceExt;

        let app = axum::Router::new()
            .route(
                "/raw-err",
                axum::routing::get(|| async { (StatusCode::BAD_REQUEST, "plain error message") }),
            )
            .layer(middleware::from_fn(error_response_formatter));

        let req = Request::builder().uri("/raw-err").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(bytes, "plain error message");
    }
    #[tokio::test]
    async fn test_error_response_formatter_preserves_custom_headers() {
        use axum::middleware;
        use axum::response::IntoResponse;
        use tower::ServiceExt;
        let app = axum::Router::new()
            .route(
                "/custom-err",
                axum::routing::get(|| async {
                    let mut resp = (StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "error": "test" }))).into_response();
                    resp.headers_mut().insert("x-custom-test", axum::http::HeaderValue::from_static("survives"));
                    resp
                }),
            )
            .layer(middleware::from_fn(error_response_formatter));

        let req = Request::builder().uri("/custom-err").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(resp.headers().get("x-custom-test").unwrap(), "survives");
    }
}
