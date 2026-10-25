pub mod languages;
pub mod plugins;
pub mod system;
pub mod translate;

use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;

/// Unified API error type. Handlers return faithful status codes; the error
/// formatter middleware only decorates the body with the request ID.
#[derive(Debug)]
pub enum ApiError {
    /// 400 — malformed input (bad field type, oversized batch, ...).
    BadRequest(String),
    /// 401 — missing or invalid API token.
    Unauthorized,
    /// 404 — resource or model not found.
    NotFound(String),
    /// 500 — unexpected failure in the translation pipeline.
    Internal(String),
}
#[derive(Serialize)]
struct ApiErrorBody<'a> {
    error: &'a str,
    message: &'a str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, std::borrow::Cow::Owned(msg)),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                std::borrow::Cow::Borrowed("Unauthorized"),
            ),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, std::borrow::Cow::Owned(msg)),
            ApiError::Internal(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, std::borrow::Cow::Owned(msg))
            }
        };
        if status == StatusCode::UNAUTHORIZED {
            (
                status,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                crate::middleware::UNAUTHORIZED_BODY,
            )
                .into_response()
        } else {
            let s = message.as_ref();
            (status, Json(ApiErrorBody { error: s, message: s })).into_response()
        }
    }
}

/// Create the application router with all routes and global middleware.
pub fn create_router(cfg: &crate::config::Config) -> Router {
    let public_routes = Router::new()
        .route("/version", get(self::system::get_version))
        .route("/health", get(self::system::get_health))
        .route("/_heartbeat__", get(self::system::get_health));
    let protected_routes = Router::new()
        .route("/languages", get(self::languages::get_languages))
        .route("/detect", post(self::languages::detect_language))
        .route("/translate", post(self::translate::translate))
        .route("/translate/batch", post(self::translate::translate_batch))
        .route("/imme", post(self::plugins::imme::translate))
        .route("/deepl", post(self::plugins::deepl::translate))
        .route("/deeplx", post(self::plugins::deeplx::translate))
        .route("/kiss", post(self::plugins::kiss::translate))
        .route("/hcfy", post(self::plugins::hcfy::translate))
        .route(
            "/google/language/translate/v2",
            post(self::plugins::google::translate_compat),
        )
        .route(
            "/google/translate_a/single",
            get(self::plugins::google::translate_single),
        )
        .route_layer(middleware::from_fn(crate::middleware::auth_middleware));

    let app = public_routes.merge(protected_routes);

    let app = if cfg.enable_web_ui {
        app.route("/", get(crate::ui::index)).route(
            &format!("/{}/{{*path}}", crate::version::UI_VERSION),
            get(crate::ui::versioned_assets),
        )
    } else {
        app
    };

    let app = if cfg.max_concurrency > 0 {
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(cfg.max_concurrency));
        app.layer(middleware::from_fn_with_state(
            semaphore,
            crate::middleware::concurrency_limit_middleware,
        ))
    } else {
        app
    };

    let app = if cfg.max_concurrency_per_ip > 0 {
        let limiter = std::sync::Arc::new(crate::middleware::IpConcurrencyLimiter::new(
            cfg.max_concurrency_per_ip,
        ));
        app.layer(middleware::from_fn_with_state(
            limiter,
            crate::middleware::ip_concurrency_limit_middleware,
        ))
    } else {
        app
    };

    app.layer(DefaultBodyLimit::max(cfg.max_request_body_size))
        .layer(middleware::from_fn(crate::middleware::log_requests))
        .layer(middleware::from_fn(
            crate::middleware::error_response_formatter,
        ))
        .layer(middleware::from_fn(crate::middleware::cors_middleware))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::await_holding_lock)]
    use std::path::PathBuf;
    use std::sync::Once;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use clap::Parser;
    use tower::ServiceExt;

    use super::*;
    use crate::config::{self, Cli, Config};

    static INIT_CONFIG: Once = Once::new();

    fn ensure_global_config_initialized() {
        INIT_CONFIG.call_once(|| {
            if !config::is_initialized() {
                let cli = Cli::try_parse_from(["mtranserver"]).expect("parse test cli");
                config::init(cli);
            }
        });
    }

    fn create_test_config(max_request_body_size: usize) -> Config {
        Config {
            log_level: "info".to_string(),
            config_dir: PathBuf::from("data"),
            model_dir: PathBuf::from("data/models"),
            host: "0.0.0.0".to_string(),
            port: "8989".to_string(),
            enable_web_ui: false,
            enable_offline_mode: false,
            worker_idle_timeout: 60,
            api_token: String::new(),
            log_dir: PathBuf::from("data/logs"),
            log_to_file: false,
            log_requests: false,
            max_sentence_length: 512,
            max_request_body_size,
            max_batch_size: 512,
            check_model_update: true,
            model_update_interval_secs: 0,
            cache_size: 2000,
            fullwidth_zh_punctuation: true,
            keep_models: Vec::new(),
            keep_top_n: 0,
            max_loaded_models: 0,
            max_threads: 0,
            max_concurrency: 0,
            max_concurrency_per_ip: 0,
            trust_proxy: false,
            log_retention_days: 7,
            preprocess_symbols: false,
            parse_formats: config::ParseFormats::default(),
            profile: false,
            bench_qps_input_length: 128,
        }
    }

    #[tokio::test]
    async fn test_default_body_limit_exceeded() {
        ensure_global_config_initialized();
        let cfg = create_test_config(512 * 1024);
        let app = create_router(&cfg);

        let padding = "a".repeat(600 * 1024);
        let payload = serde_json::json!({ "text": padding }).to_string();

        let req_detect = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from(payload.clone()))
            .unwrap();
        let resp_detect = app.clone().oneshot(req_detect).await.unwrap();
        assert_eq!(resp_detect.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let req_translate = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp_translate = app.oneshot(req_translate).await.unwrap();
        assert_eq!(resp_translate.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_default_body_limit_normal_request() {
        ensure_global_config_initialized();
        let cfg = create_test_config(512 * 1024);
        let app = create_router(&cfg);

        let payload = serde_json::json!({ "text": "hello" }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_ne!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_custom_body_limit() {
        ensure_global_config_initialized();
        let cfg = create_test_config(1024);
        let app = create_router(&cfg);

        let oversized_padding = "a".repeat(2000);
        let oversized_payload = serde_json::json!({ "text": oversized_padding }).to_string();
        let req_oversized = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from(oversized_payload))
            .unwrap();
        let resp_oversized = app.clone().oneshot(req_oversized).await.unwrap();
        assert_eq!(resp_oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let small_padding = "a".repeat(500);
        let small_payload = serde_json::json!({ "text": small_padding }).to_string();
        let req_small = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from(small_payload))
            .unwrap();
        let resp_small = app.oneshot(req_small).await.unwrap();
        assert_ne!(resp_small.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_max_concurrency_limit_rejects_with_429() {
        ensure_global_config_initialized();
        let mut cfg = create_test_config(512 * 1024);
        cfg.max_concurrency = 2;
        let app = create_router(&cfg);

        let (mut tx1, rx1) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);
        let (mut tx2, rx2) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);

        let req1 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from_stream(rx1))
            .unwrap();

        let req2 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from_stream(rx2))
            .unwrap();

        let app1 = app.clone();
        let h1 = tokio::spawn(async move { app1.oneshot(req1).await.unwrap() });
        let app2 = app.clone();
        let h2 = tokio::spawn(async move { app2.oneshot(req2).await.unwrap() });

        // Allow req1 and req2 to enter the middleware and acquire both permits
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let app3 = app.clone();
        let req3 = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp3 = app3.oneshot(req3).await.unwrap();
        assert_eq!(resp3.status(), StatusCode::TOO_MANY_REQUESTS);
        let body3 = axum::body::to_bytes(resp3.into_body(), usize::MAX)
            .await
            .unwrap();
        let body3_json: serde_json::Value = serde_json::from_slice(&body3).unwrap();
        assert_eq!(
            body3_json,
            serde_json::json!({
                "error": "Too Many Requests",
                "message": "Too Many Requests",
            })
        );

        // Release first request permit by feeding valid JSON and closing the stream
        use futures::SinkExt;
        let payload = serde_json::json!({ "text": "hello" }).to_string();
        tx1.send(Ok(axum::body::Bytes::from(payload)))
            .await
            .unwrap();
        tx1.close().await.unwrap();

        let resp1 = h1.await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);

        // Now that permit 1 is released, another request can succeed
        let app4 = app.clone();
        let req4 = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp4 = app4.oneshot(req4).await.unwrap();
        assert_eq!(resp4.status(), StatusCode::OK);

        // Release second request permit
        let payload2 = serde_json::json!({ "text": "world" }).to_string();
        tx2.send(Ok(axum::body::Bytes::from(payload2)))
            .await
            .unwrap();
        tx2.close().await.unwrap();

        let resp2 = h2.await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_max_concurrency_disabled_by_default() {
        ensure_global_config_initialized();
        let cfg = create_test_config(512 * 1024);
        assert_eq!(cfg.max_concurrency, 0);
        let app = create_router(&cfg);

        let (mut tx1, rx1) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);
        let (mut tx2, rx2) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);

        let req1 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from_stream(rx1))
            .unwrap();

        let req2 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .body(Body::from_stream(rx2))
            .unwrap();

        let app1 = app.clone();
        let h1 = tokio::spawn(async move { app1.oneshot(req1).await.unwrap() });
        let app2 = app.clone();
        let h2 = tokio::spawn(async move { app2.oneshot(req2).await.unwrap() });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Concurrency is unconstrained (0), so req3 to /health completes immediately without waiting
        let app3 = app.clone();
        let req3 = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let h3 = tokio::spawn(async move { app3.oneshot(req3).await.unwrap() });

        let resp3 = tokio::time::timeout(std::time::Duration::from_millis(100), h3)
            .await
            .expect("h3 should not be blocked when max_concurrency is 0")
            .unwrap();
        assert_eq!(resp3.status(), StatusCode::OK);

        use futures::SinkExt;
        let payload = serde_json::json!({ "text": "hello" }).to_string();
        tx1.send(Ok(axum::body::Bytes::from(payload.clone())))
            .await
            .unwrap();
        tx1.close().await.unwrap();
        tx2.send(Ok(axum::body::Bytes::from(payload)))
            .await
            .unwrap();
        tx2.close().await.unwrap();

        let resp1 = h1.await.unwrap();
        let resp2 = h2.await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);
        assert_eq!(resp2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_max_concurrency_per_ip_limit() {
        ensure_global_config_initialized();
        let mut cfg = create_test_config(512 * 1024);
        cfg.max_concurrency_per_ip = 2;
        let app = create_router(&cfg);

        let (mut tx1, rx1) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);
        let (mut tx2, rx2) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);

        let req1 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .header("x-real-ip", "192.0.2.1")
            .body(Body::from_stream(rx1))
            .unwrap();

        let req2 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .header("x-real-ip", "192.0.2.1")
            .body(Body::from_stream(rx2))
            .unwrap();

        let app1 = app.clone();
        let h1 = tokio::spawn(async move { app1.oneshot(req1).await.unwrap() });
        let app2 = app.clone();
        let h2 = tokio::spawn(async move { app2.oneshot(req2).await.unwrap() });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let app3 = app.clone();
        let req3 = Request::builder()
            .uri("/health")
            .header("x-real-ip", "192.0.2.1")
            .body(Body::empty())
            .unwrap();
        let resp3 = app3.oneshot(req3).await.unwrap();
        assert_eq!(resp3.status(), StatusCode::TOO_MANY_REQUESTS);
        let body3 = axum::body::to_bytes(resp3.into_body(), usize::MAX)
            .await
            .unwrap();
        let body3_json: serde_json::Value = serde_json::from_slice(&body3).unwrap();
        assert_eq!(
            body3_json,
            serde_json::json!({
                "error": "Too Many Requests",
                "message": "Too Many Requests",
            })
        );

        let app_diff = app.clone();
        let req_diff = Request::builder()
            .uri("/health")
            .header("x-real-ip", "198.51.100.1")
            .body(Body::empty())
            .unwrap();
        let resp_diff = app_diff.oneshot(req_diff).await.unwrap();
        assert_eq!(resp_diff.status(), StatusCode::OK);

        use futures::SinkExt;
        let payload = serde_json::json!({ "text": "hello" }).to_string();
        tx1.send(Ok(axum::body::Bytes::from(payload)))
            .await
            .unwrap();
        tx1.close().await.unwrap();

        let resp1 = h1.await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);

        let app4 = app.clone();
        let req4 = Request::builder()
            .uri("/health")
            .header("x-real-ip", "192.0.2.1")
            .body(Body::empty())
            .unwrap();
        let resp4 = app4.oneshot(req4).await.unwrap();
        assert_eq!(resp4.status(), StatusCode::OK);

        let payload2 = serde_json::json!({ "text": "world" }).to_string();
        tx2.send(Ok(axum::body::Bytes::from(payload2)))
            .await
            .unwrap();
        tx2.close().await.unwrap();

        let resp2 = h2.await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_max_concurrency_per_ip_disabled_by_default() {
        ensure_global_config_initialized();
        let cfg = create_test_config(512 * 1024);
        assert_eq!(cfg.max_concurrency_per_ip, 0);
        let app = create_router(&cfg);

        let (mut tx1, rx1) =
            futures::channel::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);

        let req1 = Request::builder()
            .method("POST")
            .uri("/detect")
            .header("content-type", "application/json")
            .header("x-real-ip", "203.0.113.1")
            .body(Body::from_stream(rx1))
            .unwrap();

        let app1 = app.clone();
        let h1 = tokio::spawn(async move { app1.oneshot(req1).await.unwrap() });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let app2 = app.clone();
        let req2 = Request::builder()
            .uri("/health")
            .header("x-real-ip", "203.0.113.1")
            .body(Body::empty())
            .unwrap();
        let resp2 = app2.oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);

        use futures::SinkExt;
        let payload = serde_json::json!({ "text": "hello" }).to_string();
        tx1.send(Ok(axum::body::Bytes::from(payload)))
            .await
            .unwrap();
        tx1.close().await.unwrap();

        let resp1 = h1.await.unwrap();
        assert_eq!(resp1.status(), StatusCode::OK);
    }

    use crate::config::TEST_LOCK as CONFIG_TEST_LOCK;

    #[tokio::test]
    async fn test_max_batch_size_default_limit() {
        let _guard = CONFIG_TEST_LOCK.lock();
        ensure_global_config_initialized();
        let cfg = create_test_config(512 * 1024);
        config::set_for_testing(cfg.clone());

        let texts_512: Vec<String> = vec!["hello".to_string(); 512];
        assert!(translate::validate_batch(&texts_512).is_ok());

        let texts_513: Vec<String> = vec!["hello".to_string(); 513];
        let err = translate::validate_batch(&texts_513).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => {
                assert_eq!(msg, "Batch too large: 513 items (max 512)");
            }
            _ => panic!("expected BadRequest"),
        }

        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "zh",
            "texts": texts_513,
        })
        .to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate/batch")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_max_batch_size_custom_limit() {
        let _guard = CONFIG_TEST_LOCK.lock();
        ensure_global_config_initialized();
        let mut cfg = create_test_config(512 * 1024);
        cfg.max_batch_size = 10;
        config::set_for_testing(cfg.clone());

        let texts_10: Vec<String> = vec!["hello".to_string(); 10];
        assert!(translate::validate_batch(&texts_10).is_ok());

        let texts_11: Vec<String> = vec!["hello".to_string(); 11];
        let err = translate::validate_batch(&texts_11).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => {
                assert_eq!(msg, "Batch too large: 11 items (max 10)");
            }
            _ => panic!("expected BadRequest"),
        }

        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "zh",
            "texts": texts_11,
        })
        .to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate/batch")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        config::set_for_testing(create_test_config(512 * 1024));
    }

    #[tokio::test]
    async fn test_max_batch_size_unlimited() {
        let _guard = CONFIG_TEST_LOCK.lock();
        ensure_global_config_initialized();
        let mut cfg = create_test_config(512 * 1024);
        cfg.max_batch_size = 0;
        config::set_for_testing(cfg.clone());

        let texts_600: Vec<String> = vec!["hello".to_string(); 600];
        assert!(translate::validate_batch(&texts_600).is_ok());

        config::set_for_testing(create_test_config(512 * 1024));
    }

    #[tokio::test]
    async fn test_translate_api_parse_parameter() {
        let _guard = CONFIG_TEST_LOCK.lock();
        ensure_global_config_initialized();
        let cfg = create_test_config(512 * 1024);

        // 1. Valid body parse="html" (lowercase)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello <b>world</b>",
            "parse": "html"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 2. Valid body parse="HTML" (uppercase)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello <b>world</b>",
            "parse": "HTML"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 3. Valid body parse="md" (lowercase)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello [link](https://example.com)",
            "parse": "md"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 4. Valid body parse="MD" (uppercase)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello [link](https://example.com)",
            "parse": "MD"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 5. Valid body parse="HTML,MD"
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello <b>world</b>",
            "parse": "HTML,MD"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 6. Valid body parse=false (boolean)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello",
            "parse": false
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 7. Valid body parse=0 (number)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello",
            "parse": 0
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 8. Reject "mod" alias
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello",
            "parse": "mod"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // 9. Reject "markdown" alias
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello",
            "parse": "markdown"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // 10. Reject arbitrary invalid syntax
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello",
            "parse": "unknown_syntax"
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // 11. Reject oversized string injection attempt (> 64 bytes)
        let app = create_router(&cfg);
        let payload = serde_json::json!({
            "from": "en",
            "to": "en",
            "text": "Hello",
            "parse": "a".repeat(1000)
        }).to_string();
        let req = Request::builder()
            .method("POST")
            .uri("/translate")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
