use std::borrow::Cow;
use axum::Json;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use crate::api::plugins::{deserialize_null_default, parse_source_lang, parse_target_lang};
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;

#[derive(Deserialize)]
pub struct DeeplxRequest {
    pub text: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub source_lang: String,
    pub target_lang: String,
}

#[derive(Serialize)]
pub struct DeeplxResponse<'a> {
    pub alternatives: &'a [&'a str],
    pub code: i32,
    pub data: String,
    pub id: i64,
    pub method: &'static str,
    pub source_lang: Cow<'static, str>,
    pub target_lang: Cow<'static, str>,
}

pub async fn translate(
    Json(body): Json<DeeplxRequest>,
) -> Result<Json<DeeplxResponse<'static>>, ApiError> {
    validate_batch(std::slice::from_ref(&body.text))?;
    let source_lang = parse_source_lang(Some(&body.source_lang));
    let target_lang = parse_target_lang(&body.target_lang, "target_lang")?;

    match service::translate_with_pivot_detected(&source_lang, &target_lang, &body.text).await {
        Ok((data, detected)) => {
            let id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| (d.as_nanos() % 10_000_000_000) as i64)
                .unwrap_or(0);
            let effective_source = if !detected.is_empty() {
                Cow::Owned(detected)
            } else if source_lang != "auto" {
                source_lang
            } else {
                Cow::Borrowed("auto")
            };
            let source_lang_deepl = super::deepl::convert_bcp47_to_deepl(&effective_source);
            let target_lang_deepl = super::deepl::convert_bcp47_to_deepl(&target_lang);
            Ok(Json(DeeplxResponse {
                alternatives: &[],
                code: 200,
                data,
                id,
                method: "Free",
                source_lang: source_lang_deepl,
                target_lang: target_lang_deepl,
            }))
        }
        Err(e) => Err(e.into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deeplx_request_deserialization_defaults() {
        let json_missing_src = r#"{"text":"hello","target_lang":"ZH"}"#;
        let req: DeeplxRequest = serde_json::from_str(json_missing_src).unwrap();
        assert_eq!(req.source_lang, "");

        let json_null_src = r#"{"text":"hello","source_lang":null,"target_lang":"ZH"}"#;
        let req2: DeeplxRequest = serde_json::from_str(json_null_src).unwrap();
        assert_eq!(req2.source_lang, "");
    }
    #[test]
    fn test_deeplx_response_echoes_detected_language() {
        let detected = "zh-Hans";
        let source_lang = "auto";
        let effective_source = if (source_lang == "auto" || source_lang.is_empty()) && !detected.is_empty() {
            detected
        } else {
            source_lang
        };
        let source_lang_deepl = crate::api::plugins::deepl::convert_bcp47_to_deepl(effective_source);
        let resp = DeeplxResponse {
            alternatives: &[],
            code: 200,
            data: "你好".to_string(),
            id: 12345,
            method: "Free",
            source_lang: source_lang_deepl,
            target_lang: Cow::Borrowed("EN"),
        };
        let serialized = serde_json::to_string(&resp).unwrap();
        assert!(serialized.contains(r#""source_lang":"ZH""#));
        assert!(!serialized.contains(r#""source_lang":"ZH-HANS""#));
    }
}
