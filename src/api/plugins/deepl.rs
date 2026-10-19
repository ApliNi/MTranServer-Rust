use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::plugins::OneOrMany;
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;
use crate::utils::lang::normalize_language_code;

#[derive(Deserialize, Debug, PartialEq)]
pub struct DeeplRequest {
    pub text: OneOrMany<String>,
    pub source_lang: Option<String>,
    pub target_lang: String,
    pub tag_handling: Option<String>,
}

#[derive(Serialize)]
pub struct DeeplTranslation {
    pub detected_source_language: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct DeeplResponse {
    pub translations: Vec<DeeplTranslation>,
}

pub async fn translate(Json(body): Json<DeeplRequest>) -> Result<Json<DeeplResponse>, ApiError> {
    validate_batch(body.text.as_slice())?;
    let source_lang = body
        .source_lang
        .as_deref()
        .map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                "auto".to_string()
            } else {
                normalize_language_code(trimmed)
            }
        })
        .unwrap_or_else(|| "auto".to_string());
    // An empty target language would build a meaningless engine key.
    let target_lang = normalize_language_code(&body.target_lang);
    if target_lang.is_empty() {
        return Err(ApiError::BadRequest(
            "Missing 'target_lang' language".to_string(),
        ));
    }

    let parse_formats = body.tag_handling.as_deref().and_then(|th| {
        if th.eq_ignore_ascii_case("html") || th.eq_ignore_ascii_case("xml") {
            Some(crate::config::ParseFormats {
                html: true,
                md: false,
            })
        } else {
            None
        }
    });

    let results = service::translate_batch_with_pivot_formats(
        &source_lang,
        &target_lang,
        body.text.as_slice(),
        parse_formats,
    )
    .await?;

    let mut translations = Vec::with_capacity(results.len());
    for (translated, effective_from) in results {
        let detected_source_language = convert_bcp47_to_deepl(&effective_from);
        translations.push(DeeplTranslation {
            detected_source_language,
            text: translated,
        });
    }

    Ok(Json(DeeplResponse { translations }))
}

pub(crate) fn convert_bcp47_to_deepl(lang: &str) -> String {
    match lang {
        "no" | "nb" => "NB".to_string(),
        "nn" => "NN".to_string(),
        "zh-CN" => "ZH-CN".to_string(),
        "zh-Hans" | "zh" => "ZH".to_string(),
        "zh-Hant" | "zh-TW" | "zh-HK" => "ZH-TW".to_string(),
        _ => lang.to_uppercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_bcp47_to_deepl() {
        assert_eq!(convert_bcp47_to_deepl("zh-Hans"), "ZH");
        assert_eq!(convert_bcp47_to_deepl("zh-CN"), "ZH-CN");
        assert_eq!(convert_bcp47_to_deepl("no"), "NB");
        assert_eq!(convert_bcp47_to_deepl("en"), "EN");
        assert_eq!(convert_bcp47_to_deepl("zh-Hant"), "ZH-TW");
        assert_eq!(convert_bcp47_to_deepl("zh-TW"), "ZH-TW");
        assert_eq!(convert_bcp47_to_deepl("nb"), "NB");
    }

    #[test]
    fn test_deepl_response_serialization() {
        let resp = DeeplResponse {
            translations: vec![DeeplTranslation {
                detected_source_language: "EN".to_string(),
                text: "Hallo".to_string(),
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            json,
            "{\"translations\":[{\"detected_source_language\":\"EN\",\"text\":\"Hallo\"}]}"
        );
    }

    #[test]
    fn test_deepl_request_deserialization_single_and_array() {
        let json_single = r#"{"text":"Hello world","target_lang":"ZH"}"#;
        let req_single: DeeplRequest = serde_json::from_str(json_single).unwrap();
        assert_eq!(req_single.text, OneOrMany::One("Hello world".to_string()));
        assert_eq!(req_single.target_lang, "ZH");

        let json_array = r#"{"text":["Hello","world"],"target_lang":"ZH"}"#;
        let req_array: DeeplRequest = serde_json::from_str(json_array).unwrap();
        assert_eq!(
            req_array.text,
            OneOrMany::Many(vec!["Hello".to_string(), "world".to_string()])
        );
    }
}
