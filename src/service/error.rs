//! Service error types for translation orchestration and engine pool management.

use std::fmt;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::ApiError;
use crate::engine::error::EngineError;

/// Strongly-typed domain error for translation service operations.
#[derive(Debug, Clone)]
pub enum ServiceError {
    /// Model file or language pair not found in registry / disk.
    ModelNotFound(String),

    /// Requested language code is not supported.
    LanguageNotSupported(String),

    /// Internal engine error (model loading, computation, shortlist, vocab).
    Engine(EngineError),

    /// Operation timed out.
    Timeout(String),

    /// Language detection failed on source text.
    LanguageDetectionFailed(String),

    /// Transient error (e.g. download or network failure) that is retryable.
    Transient(String),

    /// Internal service error or panic.
    Internal(String),
}

impl ServiceError {
    /// Whether this error represents a transient failure that may be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(self, ServiceError::Transient(_) | ServiceError::Timeout(_))
    }

    /// Construct a retryable transient error.
    pub fn transient(msg: impl Into<String>) -> Self {
        ServiceError::Transient(msg.into())
    }

    /// Construct a permanent internal error.
    pub fn permanent(msg: impl Into<String>) -> Self {
        ServiceError::Internal(msg.into())
    }

    /// Maps domain service error to corresponding HTTP status code.
    pub fn status_code(&self) -> StatusCode {
        match self {
            ServiceError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ServiceError::LanguageNotSupported(_) => StatusCode::BAD_REQUEST,
            ServiceError::LanguageDetectionFailed(_) => StatusCode::BAD_REQUEST,
            ServiceError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            ServiceError::Transient(_) => StatusCode::SERVICE_UNAVAILABLE,
            ServiceError::Engine(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ServiceError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Return the error message description.
    pub fn error_message(&self) -> String {
        self.to_string()
    }

    /// Convert into HTTP status code and error message pair.
    pub fn to_http_response(&self) -> (StatusCode, String) {
        (self.status_code(), self.error_message())
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceError::ModelNotFound(msg) => {
                if msg.starts_with("No model found") || msg.starts_with("Model not found") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "Model not found: {msg}")
                }
            }
            ServiceError::LanguageNotSupported(msg) => {
                if msg.starts_with("Language not supported") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "Language not supported: {msg}")
                }
            }
            ServiceError::Engine(err) => write!(f, "Engine error: {err}"),
            ServiceError::Timeout(msg) => write!(f, "Timeout: {msg}"),
            ServiceError::LanguageDetectionFailed(msg) => write!(f, "{msg}"),
            ServiceError::Transient(msg) => write!(f, "Transient error: {msg}"),
            ServiceError::Internal(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ServiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ServiceError::Engine(err) => Some(err),
            _ => None,
        }
    }
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.error_message();
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

impl From<EngineError> for ServiceError {
    fn from(err: EngineError) -> Self {
        ServiceError::Engine(err)
    }
}

impl From<std::io::Error> for ServiceError {
    fn from(err: std::io::Error) -> Self {
        ServiceError::Engine(EngineError::from(err))
    }
}

impl From<String> for ServiceError {
    fn from(s: String) -> Self {
        ServiceError::Internal(s)
    }
}

impl From<&str> for ServiceError {
    fn from(s: &str) -> Self {
        ServiceError::Internal(s.to_string())
    }
}

impl From<ServiceError> for String {
    fn from(err: ServiceError) -> Self {
        err.to_string()
    }
}

impl From<ServiceError> for ApiError {
    fn from(err: ServiceError) -> Self {
        match err {
            ServiceError::LanguageNotSupported(msg) => ApiError::BadRequest(msg),
            ServiceError::LanguageDetectionFailed(msg) => ApiError::BadRequest(msg),
            _ => ApiError::Internal(err.to_string()),
        }
    }
}
