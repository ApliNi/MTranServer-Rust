use axum::Json;
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
pub struct ImmeRequest {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub source_lang: String,
    pub target_lang: String,
    pub text_list: Vec<String>,
}

#[derive(Serialize)]
pub struct ImmeTranslation {
    pub detected_source_lang: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct ImmeResponse {
    pub translations: Vec<ImmeTranslation>,
}

pub async fn translate(Json(body): Json<ImmeRequest>) -> Result<Json<ImmeResponse>, ApiError> {
    validate_batch(&body.text_list)?;

    // An empty target language would build a meaningless engine key.
    let target_lang = normalize_language_code(&body.target_lang);
    if target_lang.is_empty() {
        return Err(ApiError::BadRequest(
            "Missing 'target_lang' language".to_string(),
        ));
    }

    let raw_source = body.source_lang.trim();
    let source_lang = if raw_source.is_empty() || raw_source.eq_ignore_ascii_case("auto") {
        "auto".to_string()
    } else {
        normalize_language_code(raw_source)
    };

    let results = service::translate_batch_with_pivot_formats(
        &source_lang,
        &target_lang,
        &body.text_list,
        None,
    )
    .await?;

    let translations = results
        .into_iter()
        .map(|(text, detected)| {
            let detected_source_lang = if !detected.is_empty() {
                detected
            } else if source_lang != "auto" {
                source_lang.clone()
            } else {
                "auto".to_string()
            };
            ImmeTranslation {
                detected_source_lang,
                text,
            }
        })
        .collect();
    Ok(Json(ImmeResponse { translations }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imme_response_serialization() {
        let resp = ImmeResponse {
            translations: vec![ImmeTranslation {
                detected_source_lang: "en".to_string(),
                text: "你好".to_string(),
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            json,
            "{\"translations\":[{\"detected_source_lang\":\"en\",\"text\":\"你好\"}]}"
        );
    }

    #[test]
    fn test_imme_request_deserialization_defaults() {
        let json_missing = r#"{"target_lang":"zh-Hans","text_list":["hello"]}"#;
        let req: ImmeRequest = serde_json::from_str(json_missing).unwrap();
        assert_eq!(req.source_lang, "");

        let json_null = r#"{"source_lang":null,"target_lang":"zh-Hans","text_list":["hello"]}"#;
        let req2: ImmeRequest = serde_json::from_str(json_null).unwrap();
        assert_eq!(req2.source_lang, "");
    }
}
