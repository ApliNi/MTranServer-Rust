use std::borrow::Cow;
use axum::{extract::Query, Json};
use serde::{Deserialize, Serialize};

use crate::api::plugins::{parse_source_lang, parse_target_lang, OneOrMany};
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;

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

#[derive(Serialize)]
pub struct GoogleSentence(
    pub String,
    pub String,
    pub Option<()>,
    pub Option<()>,
    pub u8,
);

#[derive(Serialize)]
pub struct GoogleSingleResponse(
    pub [GoogleSentence; 1],
    pub Option<()>,
    pub String,
    pub Option<()>,
    pub Option<()>,
    pub Option<()>,
    pub Option<()>,
    pub [(); 0],
);

pub async fn translate_compat(
    Json(body): Json<GoogleRequest>,
) -> Result<Json<GoogleResponse>, ApiError> {
    validate_batch(body.q.as_slice())?;

    let source = parse_source_lang(Some(&body.source));
    let target = parse_target_lang(&body.target, "target")?;

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

    let fallback_source = if !body.source.trim().is_empty() {
        body.source.as_str()
    } else {
        "auto"
    };

    let mut translations = Vec::with_capacity(results.len());
    for (translated, detected) in results {
        let detected_lang = if source == "auto" && !detected.is_empty() {
            convert_bcp47_to_google(&detected).to_string()
        } else {
            fallback_source.to_string()
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
) -> Result<Json<GoogleSingleResponse>, ApiError> {
    validate_batch(std::slice::from_ref(&query.q))?;
    let target = crate::utils::lang::normalize_language_code_cow(query.tl.trim());
    if target.is_empty() {
        return Err(ApiError::BadRequest("Missing target language".to_string()));
    }
    let source = parse_source_lang(query.sl.as_deref());
    let (translated, detected_source) =
        service::translate_with_pivot_detected(&source, &target, &query.q).await?;

    let effective_source = if source == "auto" && !detected_source.is_empty() {
        Cow::Owned(detected_source)
    } else {
        source
    };
    let detected_lang = convert_bcp47_to_google(&effective_source).to_string();
    Ok(Json(GoogleSingleResponse(
        [GoogleSentence(translated, query.q, None, None, 1)],
        None,
        detected_lang,
        None,
        None,
        None,
        None,
        [],
    )))
}

pub(crate) fn convert_bcp47_to_google(lang: &str) -> &str {
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

    #[test]
    fn test_google_single_response_serialization() {
        let translated = "你好";
        let original = "hello";
        let detected_lang = "en";
        let resp = GoogleSingleResponse(
            [GoogleSentence(translated.to_string(), original.to_string(), None, None, 1)],
            None,
            detected_lang.to_string(),
            None,
            None,
            None,
            None,
            [],
        );
        let json = serde_json::to_string(&resp).unwrap();
        let expected = serde_json::to_string(&serde_json::json!([
            [[translated, original, null, null, 1]],
            null,
            detected_lang,
            null,
            null,
            null,
            null,
            []
        ]))
        .unwrap();
        assert_eq!(json, expected);
    }
}
