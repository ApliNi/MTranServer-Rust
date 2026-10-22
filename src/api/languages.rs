use std::sync::Arc;
use axum::Json;
use serde::{Deserialize, Serialize};

pub use crate::models::records::{LanguageListResponse, LanguagePair};
use crate::service::detector;

#[derive(Deserialize)]
pub struct DetectRequest {
    pub text: String,
    #[serde(rename = "minConfidence")]
    pub min_confidence: Option<f32>,
}

#[derive(Serialize)]
pub struct DetectResponse {
    pub language: String,
    /// Absent (rather than `null`) when no confidence is available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

pub async fn get_languages() -> Json<Arc<LanguageListResponse>> {
    Json(crate::models::get_language_list_response())
}

pub async fn detect_language(Json(body): Json<DetectRequest>) -> Json<DetectResponse> {
    if let Some(min_conf) = body.min_confidence {
        let (language, confidence) =
            detector::detect_language_with_confidence(&body.text, min_conf).await;
        Json(DetectResponse {
            language,
            confidence: Some(confidence),
        })
    } else {
        let language = detector::detect_language(&body.text).await;
        Json(DetectResponse {
            language,
            confidence: None,
        })
    }
}
