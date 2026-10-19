use axum::Json;
use serde::{Deserialize, Serialize};

use crate::models;
use crate::service::detector;

#[derive(Serialize)]
pub struct LanguagePair {
    pub from: String,
    pub to: String,
}

#[derive(Serialize)]
pub struct LanguageListResponse {
    pub languages: Vec<String>,
    pub pairs: Vec<LanguagePair>,
}

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

pub async fn get_languages() -> Json<LanguageListResponse> {
    let languages = models::get_supported_languages();
    let pair_strings = models::get_language_pairs();
    let pairs: Vec<LanguagePair> = pair_strings
        .iter()
        .filter_map(|p| {
            let parts: Vec<&str> = p.split('_').collect();
            if parts.len() == 2 {
                Some(LanguagePair {
                    from: parts[0].to_string(),
                    to: parts[1].to_string(),
                })
            } else {
                None
            }
        })
        .collect();

    Json(LanguageListResponse { languages, pairs })
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
