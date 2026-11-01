use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::plugins::{parse_source_lang, parse_target_lang};
use crate::api::ApiError;
use crate::service;

/// Batch limits: reject oversized batches before touching the engines.
pub const MAX_BATCH_TOTAL_CHARS: usize = 65536;

/// Validate batch size and cumulative character count; empty strings are
/// allowed and short-circuit to an empty translation downstream.
pub fn validate_batch<T: AsRef<str>>(texts: &[T]) -> Result<(), ApiError> {
    let max_batch_size = if crate::config::is_initialized() {
        crate::config::get().max_batch_size
    } else {
        512
    };
    if max_batch_size > 0 && texts.len() > max_batch_size {
        return Err(ApiError::BadRequest(format!(
            "Batch too large: {} items (max {})",
            texts.len(),
            max_batch_size
        )));
    }
    let total_bytes: usize = texts.iter().map(|t| t.as_ref().len()).sum();
    if total_bytes > MAX_BATCH_TOTAL_CHARS {
        let mut total = 0;
        for t in texts {
            total += t.as_ref().chars().count();
            if total > MAX_BATCH_TOTAL_CHARS {
                return Err(ApiError::BadRequest(format!(
                    "Batch too large: {} characters (max {})",
                    total, MAX_BATCH_TOTAL_CHARS
                )));
            }
        }
    }
    Ok(())
}

#[derive(Deserialize)]
pub struct TranslateRequest {
    pub from: String,
    pub to: String,
    pub text: String,
    #[serde(default)]
    pub parse: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct TranslateResponse {
    /// Translated text, either newly generated or zero-copy shared from cache.
    pub result: crate::service::TranslatedText,
    /// Detected source language when the request used "auto"; omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    /// Handling time of the translation itself in whole milliseconds (excludes model loading).
    pub time_cost: u64,
}

#[derive(Deserialize)]
pub struct TranslateBatchRequest {
    pub from: String,
    pub to: String,
    pub texts: Vec<String>,
    #[serde(default)]
    pub parse: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct TranslateBatchResponse {
    pub results: Vec<String>,
    /// Per-item detected source language when the request used "auto"; omitted
    /// otherwise. Aligned with `results`; undetectable (e.g. empty) items are null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_languages: Option<Vec<Option<String>>>,
}

fn resolve_parse_markup(
    body_parse: Option<&serde_json::Value>,
) -> Result<Option<bool>, ApiError> {
    match body_parse {
        None => Ok(None),
        Some(val) if val.is_null() => Ok(None),
        Some(val) => crate::config::parse_markup_from_json_value(val)
            .map(Some)
            .map_err(ApiError::InvalidParameter),
    }
}

#[axum::debug_handler]
pub async fn translate(
    Json(body): Json<TranslateRequest>,
) -> Result<Json<TranslateResponse>, ApiError> {
    let to = parse_target_lang(&body.to, "to")?;
    let from = parse_source_lang(Some(&body.from));
    let parse_markup = resolve_parse_markup(body.parse.as_ref())?;
    validate_batch(std::slice::from_ref(&body.text))?;
    match service::translate_with_pivot_timed_markup(&from, &to, &body.text, parse_markup).await {
        Ok(output) => {
            let detected_language = if from == "auto" {
                (!output.detected_language.is_empty()).then_some(output.detected_language)
            } else {
                None
            };
            Ok(Json(TranslateResponse {
                result: output.text,
                detected_language,
                time_cost: output.translation_duration.as_millis() as u64,
            }))
        }
        Err(e) => Err(e.into()),
    }
}

pub async fn translate_batch(
    Json(body): Json<TranslateBatchRequest>,
) -> Result<Json<TranslateBatchResponse>, ApiError> {
    let to = parse_target_lang(&body.to, "to")?;
    let from = parse_source_lang(Some(&body.from));
    let parse_markup = resolve_parse_markup(body.parse.as_ref())?;
    validate_batch(&body.texts)?;

    let results = service::translate_batch_with_pivot_markup(
        &from,
        &to,
        &body.texts,
        parse_markup,
    )
    .await?;

    let mut results_out = Vec::with_capacity(results.len());
    let mut detected_languages = if from == "auto" {
        Some(Vec::with_capacity(results.len()))
    } else {
        None
    };

    for (result, detected) in results {
        results_out.push(result);
        if let Some(detected_list) = &mut detected_languages {
            detected_list.push((!detected.is_empty()).then_some(detected));
        }
    }
    Ok(Json(TranslateBatchResponse {
        results: results_out,
        detected_languages,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_response_serialization() {
        let resp = TranslateResponse {
            result: "hello".into(),
            detected_language: None,
            time_cost: 42,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, "{\"result\":\"hello\",\"time_cost\":42}");

        let resp_with_lang = TranslateResponse {
            result: "hello".into(),
            detected_language: Some("en".to_string()),
            time_cost: 42,
        };
        let json_with_lang = serde_json::to_string(&resp_with_lang).unwrap();
        assert_eq!(
            json_with_lang,
            "{\"result\":\"hello\",\"detected_language\":\"en\",\"time_cost\":42}"
        );
    }

    #[test]
    fn test_translate_batch_response_serialization() {
        let resp = TranslateBatchResponse {
            results: vec!["hello".to_string(), "world".to_string()],
            detected_languages: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, "{\"results\":[\"hello\",\"world\"]}");

        let resp_with_lang = TranslateBatchResponse {
            results: vec!["hello".to_string(), "world".to_string()],
            detected_languages: Some(vec![Some("en".to_string()), None]),
        };
        let json_with_lang = serde_json::to_string(&resp_with_lang).unwrap();
        assert_eq!(
            json_with_lang,
            "{\"results\":[\"hello\",\"world\"],\"detected_languages\":[\"en\",null]}"
        );
    }

    #[test]
    fn test_validate_batch() {
        let valid = vec!["hello".to_string(), "world".to_string()];
        assert!(validate_batch(&valid).is_ok());

        let empty: Vec<String> = Vec::new();
        assert!(validate_batch(&empty).is_ok());

        let str_slices = ["hello", "world"];
        assert!(validate_batch(&str_slices).is_ok());

        let oversized = "a".repeat(MAX_BATCH_TOTAL_CHARS + 1);
        assert!(validate_batch(&[oversized]).is_err());
    }

    #[test]
    fn test_resolve_parse_markup() {
        // Default (None in body)
        assert_eq!(resolve_parse_markup(None).unwrap(), None);

        // Null value in body
        assert_eq!(resolve_parse_markup(Some(&serde_json::Value::Null)).unwrap(), None);

        // Booleans
        assert_eq!(resolve_parse_markup(Some(&serde_json::json!(true))).unwrap(), Some(true));
        assert_eq!(resolve_parse_markup(Some(&serde_json::json!(false))).unwrap(), Some(false));

        // Numbers
        assert_eq!(resolve_parse_markup(Some(&serde_json::json!(1))).unwrap(), Some(true));
        assert_eq!(resolve_parse_markup(Some(&serde_json::json!(0))).unwrap(), Some(false));

        // Strings enabling markup
        for val in ["html", "HTML", "md", "MD", "html,md", "HTML,MD", "true", "TRUE", "1", "all", "yes", "on"] {
            let body_str = serde_json::json!(val);
            assert_eq!(resolve_parse_markup(Some(&body_str)).unwrap(), Some(true), "Failed for {}", val);
        }

        // Strings disabling markup
        for val in ["false", "FALSE", "0", "none", "raw", "off", "no", "disable", "text"] {
            let body_str = serde_json::json!(val);
            assert_eq!(resolve_parse_markup(Some(&body_str)).unwrap(), Some(false), "Failed for {}", val);
        }

        // Invalid strings
        for invalid in ["mod", "markdown", "invalid_xyz"] {
            let body_invalid = serde_json::json!(invalid);
            assert!(resolve_parse_markup(Some(&body_invalid)).is_err(), "Should fail for {}", invalid);
        }

        // Oversized string injection attempt
        let body_oversized = serde_json::json!("a".repeat(1024));
        assert!(resolve_parse_markup(Some(&body_oversized)).is_err());

        // Invalid types and out-of-range numbers
        assert!(resolve_parse_markup(Some(&serde_json::json!(2))).is_err());
        assert!(resolve_parse_markup(Some(&serde_json::json!(-1))).is_err());
        assert!(resolve_parse_markup(Some(&serde_json::json!(0.5))).is_err());
        assert!(resolve_parse_markup(Some(&serde_json::json!(["html"]))).is_err());
        assert!(resolve_parse_markup(Some(&serde_json::json!({"html": true}))).is_err());
    }
}
