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

#[derive(Serialize, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct GoogleTranslation {
    pub translatedText: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detectedSourceLanguage: Option<String>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct GoogleTranslateData {
    pub translations: Vec<GoogleTranslation>,
}

#[derive(Serialize, Debug, PartialEq)]
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

    let static_source = if source != "auto" {
        Some(convert_bcp47_to_google(&source).to_string())
    } else {
        None
    };

    let mut translations = Vec::with_capacity(results.len());
    for (translated, detected) in results {
        let detected_lang = match &static_source {
            Some(s) => Some(s.clone()),
            None if !detected.is_empty() && detected != "auto" => {
                Some(convert_bcp47_to_google(&detected).to_string())
            }
            None => None,
        };
        translations.push(GoogleTranslation {
            translatedText: translated,
            detectedSourceLanguage: detected_lang,
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

    let effective_source = if source == "auto" {
        if detected_source.is_empty() || detected_source == "auto" {
            Cow::Borrowed("")
        } else {
            Cow::Owned(detected_source)
        }
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
        "zh-Hans" | "zh" => "zh-CN",
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
        assert_eq!(convert_bcp47_to_google("zh"), "zh-CN");
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
    #[tokio::test]
    async fn test_translate_compat_source_language_normalization() {
        // Explicit "zh" normalized to "zh-CN"
        let req_zh = GoogleRequest {
            q: OneOrMany::One("测试".to_string()),
            target: "zh".to_string(),
            source: "zh".to_string(),
            format: None,
        };
        let res_zh = translate_compat(Json(req_zh)).await.unwrap();
        assert_eq!(
            res_zh.data.translations[0].detectedSourceLanguage.as_deref(),
            Some("zh-CN")
        );

        // Explicit "zh-Hans" normalized to "zh-CN"
        let req_hans = GoogleRequest {
            q: OneOrMany::One("测试".to_string()),
            target: "zh".to_string(),
            source: "zh-Hans".to_string(),
            format: None,
        };
        let res_hans = translate_compat(Json(req_hans)).await.unwrap();
        assert_eq!(
            res_hans.data.translations[0].detectedSourceLanguage.as_deref(),
            Some("zh-CN")
        );

        // Explicit "zh-Hant" normalized to "zh-TW"
        let req_hant = GoogleRequest {
            q: OneOrMany::One("測試".to_string()),
            target: "zh-Hant".to_string(),
            source: "zh-Hant".to_string(),
            format: None,
        };
        let res_hant = translate_compat(Json(req_hant)).await.unwrap();
        assert_eq!(
            res_hant.data.translations[0].detectedSourceLanguage.as_deref(),
            Some("zh-TW")
        );

        // Auto source with empty text results in None (never "auto")
        let req_auto_empty = GoogleRequest {
            q: OneOrMany::One(String::new()),
            target: "en".to_string(),
            source: "auto".to_string(),
            format: None,
        };
        let res_auto_empty = translate_compat(Json(req_auto_empty)).await.unwrap();
        assert_eq!(
            res_auto_empty.data.translations[0].detectedSourceLanguage,
            None
        );
        let serialized = serde_json::to_string(&res_auto_empty.0).unwrap();
        assert!(!serialized.contains("detectedSourceLanguage"), "Should omit detectedSourceLanguage when None: {serialized}");
    }
}
