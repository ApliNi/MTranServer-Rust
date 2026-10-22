use std::borrow::Cow;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::plugins::{parse_source_lang, parse_target_lang, OneOrMany};
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;

#[derive(Deserialize, Debug, PartialEq)]
pub struct DeeplRequest {
    pub text: OneOrMany<String>,
    pub source_lang: Option<String>,
    pub target_lang: String,
    pub tag_handling: Option<String>,
}

#[derive(Serialize)]
pub struct DeeplTranslation {
    pub detected_source_language: Cow<'static, str>,
    pub text: String,
}

#[derive(Serialize)]
pub struct DeeplResponse {
    pub translations: Vec<DeeplTranslation>,
}

pub async fn translate(Json(body): Json<DeeplRequest>) -> Result<Json<DeeplResponse>, ApiError> {
    validate_batch(body.text.as_slice())?;
    let source_lang = parse_source_lang(body.source_lang.as_deref());
    let target_lang = parse_target_lang(&body.target_lang, "target_lang")?;

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

pub(crate) fn convert_bcp47_to_deepl(lang: &str) -> Cow<'static, str> {
    match lang {
        "no" | "nb" => Cow::Borrowed("NB"),
        "nn" => Cow::Borrowed("NN"),
        "zh-CN" => Cow::Borrowed("ZH-CN"),
        "zh-Hans" | "zh" => Cow::Borrowed("ZH"),
        "zh-Hant" | "zh-TW" | "zh-HK" => Cow::Borrowed("ZH-TW"),
        "en" => Cow::Borrowed("EN"),
        "ja" => Cow::Borrowed("JA"),
        "de" => Cow::Borrowed("DE"),
        "fr" => Cow::Borrowed("FR"),
        "es" => Cow::Borrowed("ES"),
        "it" => Cow::Borrowed("IT"),
        "pt" => Cow::Borrowed("PT"),
        "ru" => Cow::Borrowed("RU"),
        "ko" => Cow::Borrowed("KO"),
        "ar" => Cow::Borrowed("AR"),
        "tr" => Cow::Borrowed("TR"),
        "vi" => Cow::Borrowed("VI"),
        "pl" => Cow::Borrowed("PL"),
        "nl" => Cow::Borrowed("NL"),
        "el" => Cow::Borrowed("EL"),
        "th" => Cow::Borrowed("TH"),
        "hi" => Cow::Borrowed("HI"),
        "id" => Cow::Borrowed("ID"),
        "uk" => Cow::Borrowed("UK"),
        "ro" => Cow::Borrowed("RO"),
        "cs" => Cow::Borrowed("CS"),
        "sv" => Cow::Borrowed("SV"),
        "hu" => Cow::Borrowed("HU"),
        "fi" => Cow::Borrowed("FI"),
        "da" => Cow::Borrowed("DA"),
        "bg" => Cow::Borrowed("BG"),
        "he" => Cow::Borrowed("HE"),
        "hr" => Cow::Borrowed("HR"),
        "sr" => Cow::Borrowed("SR"),
        "sk" => Cow::Borrowed("SK"),
        "sl" => Cow::Borrowed("SL"),
        "lt" => Cow::Borrowed("LT"),
        "lv" => Cow::Borrowed("LV"),
        "et" => Cow::Borrowed("ET"),
        "ca" => Cow::Borrowed("CA"),
        "gl" => Cow::Borrowed("GL"),
        "eu" => Cow::Borrowed("EU"),
        "fa" => Cow::Borrowed("FA"),
        "bn" => Cow::Borrowed("BN"),
        "auto" => Cow::Borrowed("AUTO"),
        _ => Cow::Owned(lang.to_uppercase()),
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
                detected_source_language: Cow::Borrowed("EN"),
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
