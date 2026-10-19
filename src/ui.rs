//! Embedded web UI — static assets compiled into the binary.
//!
//! Assets live in `src/ui/` and are embedded at build time via `rust-embed`,
//! keeping the single-binary distribution property. Routes are wired up in
//! `main.rs` and only registered when `--ui` / `MT_ENABLE_UI` is enabled.
//!
//! Versioning: the entry HTML references assets through
//! `<base href="/${uiVersion}/">`, where `${uiVersion}` is substituted with
//! [`crate::version::UI_VERSION`] at serve time. The HTML itself is served
//! from `/` with negotiated caching (ETag), while every asset under
//! `/${uiVersion}/` is immutable for a year — bumping `UI_VERSION` changes
//! the base path, so caches invalidate naturally without revalidation.

use std::borrow::Cow;

use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode, Uri},
    response::Response,
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "src/ui/"]
struct Asset;

/// UI asset version segment: the HTML `<base href>` and the versioned asset
/// prefix both derive from [`crate::version::UI_VERSION`].
pub fn ui_version() -> &'static str {
    crate::version::UI_VERSION
}

static HTML_CACHED: std::sync::LazyLock<(String, String)> = std::sync::LazyLock::new(|| {
    let raw = match Asset::get("index.html") {
        Some(content) => match content.data {
            Cow::Borrowed(bytes) => Cow::Borrowed(std::str::from_utf8(bytes).unwrap_or_default()),
            Cow::Owned(bytes) => Cow::Owned(String::from_utf8_lossy(&bytes).into_owned()),
        },
        None => Cow::Borrowed(""),
    };
    let html = raw.replace("${uiVersion}", ui_version());
    let etag = format!("W/\"{:x}-{}\"", html.len(), ui_version());
    (html, etag)
});

/// Serve the SPA entry point at `/` with negotiated caching (ETag).
pub async fn index(headers: HeaderMap) -> Response {
    let (html, etag) = &*HTML_CACHED;
    if if_none_match_hit(&headers, etag) {
        return Response::builder()
            .status(StatusCode::NOT_MODIFIED)
            .header(header::ETAG, etag.as_str())
            .body(Body::empty())
            .unwrap();
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .header(header::ETAG, etag.as_str())
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from(html.clone()))
        .unwrap()
}

/// Serve an embedded asset under `/${uiVersion}/…` with one-year immutable
/// caching. The HTML is excluded: the entry document lives at `/` only, so
/// a stale cached `index.html` under an old version prefix can never shadow
/// a new deployment.
pub async fn versioned_assets(uri: Uri) -> Response {
    let raw = uri.path();
    let Some(path) = raw.strip_prefix(&format!("/{}/", ui_version())) else {
        return not_found();
    };
    if path.is_empty() || path == "index" || path.ends_with(".html") {
        return not_found();
    }
    serve_embedded(path)
}

fn if_none_match_hit(headers: &HeaderMap, etag: &str) -> bool {
    headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').any(|tag| tag.trim() == etag))
}


fn serve_embedded(path: &str) -> Response {
    match Asset::get(path) {
        Some(content) => {
            let body: Body = match content.data {
                // Release builds return borrowed static bytes — serve them
                Cow::Borrowed(bytes) => Body::from(bytes),
                Cow::Owned(bytes) => Body::from(bytes),
            };
            // The versioned prefix already pins the content: a rebuild that
            // changes any asset also changes `ui_version()`, so clients can
            // cache aggressively without revalidation.
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime_for(path))
                .header(header::CACHE_CONTROL, "public, max-age=31536000, immutable")
                .body(body)
                .unwrap()
        }
        None => not_found(),
    }
}

fn not_found() -> Response {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from("404"))
        .unwrap()
}

fn mime_for(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".json") {
        "application/json; charset=utf-8"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".ico") {
        "image/x-icon"
    } else if path.ends_with(".woff2") {
        "font/woff2"
    } else {
        "application/octet-stream"
    }
}
