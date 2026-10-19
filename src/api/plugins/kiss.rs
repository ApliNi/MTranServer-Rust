use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;
use crate::utils::lang::normalize_language_code_cow;

fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Deserialize)]
pub struct KissRequest {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub from: String,
    pub to: String,
    pub text: Option<String>,
    pub texts: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct KissTranslation {
    pub text: String,
    pub src: String,
}

#[derive(Serialize)]
pub struct KissResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translations: Option<Vec<KissTranslation>>,
}

pub async fn translate(Json(body): Json<KissRequest>) -> Result<Json<KissResponse>, ApiError> {
    let raw_from = body.from.trim();
    let from = if raw_from.is_empty() || raw_from.eq_ignore_ascii_case("auto") {
        std::borrow::Cow::Borrowed("auto")
    } else {
        normalize_language_code_cow(raw_from)
    };
    // An empty target language would build a meaningless engine key.
    let to = normalize_language_code_cow(&body.to);
    if to.is_empty() {
        return Err(ApiError::BadRequest("Missing 'to' language".to_string()));
    }

    // TS only enters the texts branch when `texts` is a non-empty array.
    let texts = body.texts.as_deref().filter(|t| !t.is_empty());

    if let Some(texts) = texts {
        validate_batch(texts)?;

        let results = service::translate_batch_with_pivot_formats(&from, &to, texts, None).await?;
        let translations = results
            .into_iter()
            .map(|(translated, detected)| {
                let src = if from == "auto" && !detected.is_empty() {
                    detected
                } else if !raw_from.is_empty() {
                    body.from.clone()
                } else {
                    "auto".to_string()
                };
                KissTranslation {
                    text: translated,
                    src,
                }
            })
            .collect();

        Ok(Json(KissResponse {
            text: None,
            src: None,
            translations: Some(translations),
        }))
    } else if let Some(text) = &body.text {
        validate_batch(std::slice::from_ref(text))?;
        let results = service::translate_batch_with_pivot_formats(
            &from,
            &to,
            std::slice::from_ref(text),
            None,
        )
        .await?;
        let (translated, detected) = results.into_iter().next().unwrap_or_default();
        let src = if from == "auto" && !detected.is_empty() {
            detected
        } else if !raw_from.is_empty() {
            body.from.clone()
        } else {
            "auto".to_string()
        };
        Ok(Json(KissResponse {
            text: Some(translated),
            src: Some(src),
            translations: None,
        }))
    } else {
        Err(ApiError::BadRequest(
            "Missing text or texts field".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kiss_request_deserialization_defaults() {
        let json_missing = r#"{"to":"zh-Hans","text":"hello"}"#;
        let req: KissRequest = serde_json::from_str(json_missing).unwrap();
        assert_eq!(req.from, "");

        let json_null = r#"{"from":null,"to":"zh-Hans","texts":["hello","world"]}"#;
        let req2: KissRequest = serde_json::from_str(json_null).unwrap();
        assert_eq!(req2.from, "");
    }
}
