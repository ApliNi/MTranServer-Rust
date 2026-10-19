use axum::{extract::Query, Json};
use serde::{Deserialize, Serialize};

use crate::api::plugins::OneOrMany;
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;
use crate::utils::lang::normalize_language_code;

#[derive(Deserialize, Debug, PartialEq)]
pub struct GoogleRequest {
    pub q: OneOrMany<String>,
    #[serde(default)]
    pub source: String,
    pub target: String,
    pub format: Option<String>,
}

#[derive(Serialize)]
#[allow(non_snake_case)]
pub struct GoogleTranslation {
    pub translatedText: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detectedSourceLanguage: Option<String>,
}

#[derive(Serialize)]
pub struct GoogleTranslateData {
    pub translations: Vec<GoogleTranslation>,
}

#[derive(Serialize)]
pub struct GoogleResponse {
    pub data: GoogleTranslateData,
}

#[derive(Deserialize)]
pub struct GoogleSingleQuery {
    pub sl: Option<String>,
    pub tl: String,
    pub q: String,
}

pub async fn translate_compat(
    Json(body): Json<GoogleRequest>,
) -> Result<Json<GoogleResponse>, ApiError> {
    validate_batch(body.q.as_slice())?;

    let raw_source = body.source.trim();
    let source = if raw_source.is_empty() || raw_source.eq_ignore_ascii_case("auto") {
        "auto".to_string()
    } else {
        normalize_language_code(raw_source)
    };
    let detected_source = body.source.clone();
    // An empty target language would build a meaningless engine key.
    let target = normalize_language_code(&body.target);
    if target.is_empty() {
        return Err(ApiError::BadRequest(
            "Missing 'target' language".to_string(),
        ));
    }

    let parse_formats = body.format.as_deref().and_then(|f| {
        if f.eq_ignore_ascii_case("html") {
            Some(crate::config::ParseFormats {
                html: true,
                md: false,
            })
        } else if f.eq_ignore_ascii_case("text") {
            Some(crate::config::ParseFormats::NONE)
        } else {
            None
        }
    });

    let results = service::translate_batch_with_pivot_formats(
        &source,
        &target,
        body.q.as_slice(),
        parse_formats,
    )
    .await?;

    let mut translations = Vec::with_capacity(results.len());
    for (translated, detected) in results {
        let detected_lang = if source == "auto" && !detected.is_empty() {
            convert_bcp47_to_google(&detected).to_string()
        } else if !detected_source.is_empty() {
            detected_source.clone()
        } else {
            "auto".to_string()
        };
        translations.push(GoogleTranslation {
            translatedText: translated,
            detectedSourceLanguage: Some(detected_lang),
        });
    }

    Ok(Json(GoogleResponse {
        data: GoogleTranslateData { translations },
    }))
}

pub async fn translate_single(
    Query(query): Query<GoogleSingleQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_batch(std::slice::from_ref(&query.q))?;
    let target = normalize_language_code(&query.tl);
    if target.is_empty() {
        return Err(ApiError::BadRequest("Missing target language".to_string()));
    }
    let raw_source = query.sl.as_deref().unwrap_or("auto").trim();
    let source = if raw_source.is_empty() || raw_source.eq_ignore_ascii_case("auto") {
        "auto".to_string()
    } else {
        normalize_language_code(raw_source)
    };
    let (translated, detected_source) =
        service::translate_with_pivot_detected(&source, &target, &query.q).await?;

    let effective_source = if source == "auto" && !detected_source.is_empty() {
        detected_source
    } else {
        source
    };
    let detected_lang = convert_bcp47_to_google(&effective_source);

    // Google format: [[[translated, original, null, null, 1]], null, detectedLang, ...]
    Ok(Json(serde_json::json!([
        [[translated, query.q, null, null, 1]],
        null,
        detected_lang,
        null,
        null,
        null,
        null,
        []
    ])))
}

fn convert_bcp47_to_google(lang: &str) -> &str {
    match lang {
        "zh-Hans" => "zh-CN",
        "zh-Hant" => "zh-TW",
        _ => lang,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_bcp47_to_google() {
        assert_eq!(convert_bcp47_to_google("zh-Hans"), "zh-CN");
        assert_eq!(convert_bcp47_to_google("zh-Hant"), "zh-TW");
        assert_eq!(convert_bcp47_to_google("en"), "en");
    }

    #[test]
    fn test_google_response_serialization() {
        let resp = GoogleResponse {
            data: GoogleTranslateData {
                translations: vec![GoogleTranslation {
                    translatedText: "Hello".to_string(),
                    detectedSourceLanguage: Some("zh-CN".to_string()),
                }],
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            json,
            "{\"data\":{\"translations\":[{\"translatedText\":\"Hello\",\"detectedSourceLanguage\":\"zh-CN\"}]}}"
        );
    }

    #[test]
    fn test_google_request_deserialization_optional_source() {
        let json = r#"{"q":"Hello world","target":"zh"}"#;
        let req: GoogleRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.source, "");
        assert_eq!(req.target, "zh");
        assert_eq!(req.q, OneOrMany::One("Hello world".to_string()));
    }

    #[test]
    fn test_google_request_deserialization_array_q() {
        let json = r#"{"q":["Hello","world"],"target":"zh"}"#;
        let req: GoogleRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.source, "");
        assert_eq!(req.target, "zh");
        assert_eq!(
            req.q,
            OneOrMany::Many(vec!["Hello".to_string(), "world".to_string()])
        );
    }
}
