use axum::Json;
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;
use crate::utils::lang::normalize_language_code;
fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Deserialize)]
pub struct DeeplxRequest {
    pub text: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub source_lang: String,
    pub target_lang: String,
}

#[derive(Serialize)]
pub struct DeeplxResponse {
    pub alternatives: Vec<String>,
    pub code: i32,
    pub data: String,
    pub id: i64,
    pub method: String,
    pub source_lang: String,
    pub target_lang: String,
}

pub async fn translate(Json(body): Json<DeeplxRequest>) -> Result<Json<DeeplxResponse>, ApiError> {
    validate_batch(std::slice::from_ref(&body.text))?;
    let raw_source = body.source_lang.trim();
    let source_lang = if raw_source.is_empty() || raw_source.eq_ignore_ascii_case("auto") {
        "auto".to_string()
    } else {
        normalize_language_code(raw_source)
    };
    // An empty target language would build a meaningless engine key.
    let target_lang = normalize_language_code(&body.target_lang);
    if target_lang.is_empty() {
        return Err(ApiError::BadRequest(
            "Missing 'target_lang' language".to_string(),
        ));
    }

    match service::translate_with_pivot_detected(&source_lang, &target_lang, &body.text).await {
        Ok((data, detected)) => {
            let id: i64 = rand::thread_rng().gen_range(0..10000000000);
            let effective_source = if !detected.is_empty() {
                detected
            } else if source_lang != "auto" {
                source_lang
            } else {
                "auto".to_string()
            };
            let source_lang_deepl = super::deepl::convert_bcp47_to_deepl(&effective_source);
            let target_lang_deepl = super::deepl::convert_bcp47_to_deepl(&target_lang);
            Ok(Json(DeeplxResponse {
                alternatives: vec![],
                code: 200,
                data,
                id,
                method: "Free".to_string(),
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
            alternatives: vec![],
            code: 200,
            data: "你好".to_string(),
            id: 12345,
            method: "Free".to_string(),
            source_lang: source_lang_deepl,
            target_lang: "EN".to_string(),
        };
        let serialized = serde_json::to_string(&resp).unwrap();
        assert!(serialized.contains(r#""source_lang":"ZH""#));
        assert!(!serialized.contains(r#""source_lang":"ZH-HANS""#));
    }
}
