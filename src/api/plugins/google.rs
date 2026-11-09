use axum::{extract::Query, Json};
use serde::{Deserialize, Serialize};

use crate::api::plugins::{deserialize_null_default, parse_target_lang, OneOrMany};
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;

#[derive(Deserialize, Debug, PartialEq)]
pub struct GoogleRequest {
    pub q: OneOrMany<String>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
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
    let target = parse_target_lang(&body.target, "target")?;
    let parse_markup = resolve_google_format(body.format.as_deref());

    let results = service::translate_batch_with_pivot_markup(
        "auto",
        &target,
        body.q.as_slice(),
        parse_markup,
    )
    .await?;

    let mut translations = Vec::with_capacity(results.len());
    for (translated, detected) in results {
        let detected_lang = if !detected.is_empty() && detected != "auto" {
            Some(convert_bcp47_to_google(&detected).to_string())
        } else {
            None
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
    let (translated, detected_source) =
        service::translate_with_pivot_detected("auto", &target, &query.q).await?;

    let effective_source = if !detected_source.is_empty() && detected_source != "auto" {
        detected_source.as_str()
    } else {
        ""
    };
    let detected_lang = convert_bcp47_to_google(effective_source).to_string();
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

pub(crate) fn resolve_google_format(format: Option<&str>) -> Option<bool> {
    format.and_then(|f| {
        if f.eq_ignore_ascii_case("html") {
            Some(true)
        } else if f.eq_ignore_ascii_case("text") {
            Some(false)
        } else {
            None
        }
    })
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

        let json_null = r#"{"q":"Hello world","source":null,"target":"zh"}"#;
        let req_null: GoogleRequest = serde_json::from_str(json_null).unwrap();
        assert_eq!(req_null.source, "");
        assert_eq!(req_null.target, "zh");
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
        // Even if incoming source is specified as "en", text "测试" is detected as Simplified Chinese ("zh-CN")
        let req_zh = GoogleRequest {
            q: OneOrMany::One("测试".to_string()),
            target: "zh".to_string(),
            source: "en".to_string(),
            format: None,
        };
        let res_zh = translate_compat(Json(req_zh)).await.unwrap();
        assert_eq!(
            res_zh.data.translations[0].detectedSourceLanguage.as_deref(),
            Some("zh-CN")
        );

        // Without source parameter, text "测试" is detected as Simplified Chinese ("zh-CN")
        let req_empty_source = GoogleRequest {
            q: OneOrMany::One("测试".to_string()),
            target: "zh".to_string(),
            source: String::new(),
            format: None,
        };
        let res_empty_source = translate_compat(Json(req_empty_source)).await.unwrap();
        assert_eq!(
            res_empty_source.data.translations[0].detectedSourceLanguage.as_deref(),
            Some("zh-CN")
        );

        // Even if incoming source is specified as "en", Traditional Chinese text is detected as "zh-TW"
        let req_hant = GoogleRequest {
            q: OneOrMany::One("這份報告".to_string()),
            target: "zh-Hant".to_string(),
            source: "en".to_string(),
            format: None,
        };
        let res_hant = translate_compat(Json(req_hant)).await.unwrap();
        assert_eq!(
            res_hant.data.translations[0].detectedSourceLanguage.as_deref(),
            Some("zh-TW")
        );

        // Empty text results in None for detectedSourceLanguage (never "auto")
        let req_auto_empty = GoogleRequest {
            q: OneOrMany::One(String::new()),
            target: "en".to_string(),
            source: "ja".to_string(),
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

    #[tokio::test]
    async fn test_translate_single_ignores_sl_and_detects() {
        let query = GoogleSingleQuery {
            sl: Some("en".to_string()),
            tl: "zh".to_string(),
            q: "测试".to_string(),
        };
        let res = translate_single(Query(query)).await.unwrap();
        assert_eq!(res.0 .2, "zh-CN");
    }
    #[test]
    fn test_resolve_google_format() {
        assert_eq!(resolve_google_format(Some("html")), Some(true));
        assert_eq!(resolve_google_format(Some("HTML")), Some(true));
        assert_eq!(resolve_google_format(Some("text")), Some(false));
        assert_eq!(resolve_google_format(Some("TEXT")), Some(false));
        assert_eq!(resolve_google_format(Some("other")), None);
        assert_eq!(resolve_google_format(None), None);
    }
}
