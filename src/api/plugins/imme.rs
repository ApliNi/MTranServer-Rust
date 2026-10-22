use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::plugins::{deserialize_null_default, parse_source_lang, parse_target_lang};
use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;

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

    let target_lang = parse_target_lang(&body.target_lang, "target_lang")?;
    let source_lang = parse_source_lang(Some(&body.source_lang));

    let results = service::translate_batch_with_pivot_formats(
        &source_lang,
        &target_lang,
        &body.text_list,
        None,
    )
    .await?;

    let fallback_source = if source_lang != "auto" {
        source_lang.as_ref()
    } else {
        "auto"
    };

    let translations = results
        .into_iter()
        .map(|(text, detected)| {
            let detected_source_lang = if !detected.is_empty() {
                detected
            } else {
                fallback_source.to_string()
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
