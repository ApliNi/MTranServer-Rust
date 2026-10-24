use std::collections::HashMap;
use std::sync::LazyLock;

use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::translate::validate_batch;
use crate::api::ApiError;
use crate::service;
use crate::utils::lang::normalize_language_code;

/// Hcfy language name -> BCP-47 code, built once on first use.
static HCFY_TO_BCP47: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::with_capacity(32);
    m.insert("中文(简体)", "zh-Hans");
    m.insert("中文(繁体)", "zh-Hant");
    m.insert("英语", "en");
    m.insert("日语", "ja");
    m.insert("韩语", "ko");
    m.insert("法语", "fr");
    m.insert("德语", "de");
    m.insert("西班牙语", "es");
    m.insert("俄语", "ru");
    m.insert("意大利语", "it");
    m.insert("葡萄牙语", "pt");
    m.insert("阿拉伯语", "ar");
    m.insert("越南语", "vi");
    m.insert("泰语", "th");
    m.insert("波兰语", "pl");
    m.insert("荷兰语", "nl");
    m.insert("土耳其语", "tr");
    m.insert("印地语", "hi");
    m.insert("印度尼西亚语", "id");
    m.insert("乌克兰语", "uk");
    m
});

/// BCP-47 code -> hcfy language name, built once on first use.
static BCP47_TO_HCFY: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::with_capacity(32);
    m.insert("zh-Hans", "中文(简体)");
    m.insert("zh-CN", "中文(简体)");
    m.insert("zh-Hant", "中文(繁体)");
    m.insert("zh-TW", "中文(繁体)");
    m.insert("en", "英语");
    m.insert("ja", "日语");
    m.insert("ko", "韩语");
    m.insert("fr", "法语");
    m.insert("de", "德语");
    m.insert("es", "西班牙语");
    m.insert("ru", "俄语");
    m.insert("it", "意大利语");
    m.insert("pt", "葡萄牙语");
    m.insert("ar", "阿拉伯语");
    m.insert("vi", "越南语");
    m.insert("th", "泰语");
    m.insert("pl", "波兰语");
    m.insert("nl", "荷兰语");
    m.insert("tr", "土耳其语");
    m.insert("hi", "印地语");
    m.insert("id", "印度尼西亚语");
    m.insert("uk", "乌克兰语");
    m
});

#[derive(Deserialize, Debug)]
pub struct HcfyRequest {
    /// Accepted for API compatibility; the handler never reads it.
    #[allow(dead_code)]
    pub name: Option<String>,
    pub text: String,
    pub destination: Vec<String>,
    pub source: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct HcfyResponse {
    pub text: String,
    pub from: String,
    pub to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Vec<String>>,
}

fn contains_chinese(text: &str) -> bool {
    text.chars().any(|c| matches!(c, '\u{4E00}'..='\u{9FFF}'))
}

fn contains_japanese(text: &str) -> bool {
    text.chars()
        .any(|c| matches!(c, '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}'))
}

fn contains_korean(text: &str) -> bool {
    text.chars().any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}'))
}

pub async fn translate(Json(mut body): Json<HcfyRequest>) -> Result<Json<HcfyResponse>, ApiError> {
    validate_batch(std::slice::from_ref(&body.text))?;

    let hcfy_to_bcp47 = &*HCFY_TO_BCP47;
    let bcp47_to_hcfy = &*BCP47_TO_HCFY;

    let mut source_lang = "auto".to_string();
    if let Some(src) = &body.source {
        let trimmed = src.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            source_lang = hcfy_to_bcp47
                .get(trimmed)
                .map(|&s| s.to_string())
                .unwrap_or_else(|| normalize_language_code(trimmed));
        }
    }
    let target_lang_name = match body.destination.first() {
        Some(name) if !name.trim().is_empty() => name,
        _ => {
            return Err(ApiError::BadRequest("Missing or empty 'destination' array".to_string()));
        }
    };
    let mut target_lang = hcfy_to_bcp47
        .get(target_lang_name.as_str())
        .map(|&s| s.to_string())
        .unwrap_or_else(|| normalize_language_code(target_lang_name));
    if target_lang.is_empty() {
        return Err(ApiError::BadRequest("Missing or invalid target language".to_string()));
    }

    let detected_source = if source_lang == "auto" {
        let detected = crate::service::detector::detect_language(&body.text).await;
        if !detected.is_empty() && detected != "un" {
            detected
        } else if contains_japanese(&body.text) {
            "ja".to_string()
        } else if contains_korean(&body.text) {
            "ko".to_string()
        } else if contains_chinese(&body.text) {
            crate::service::detector::distinguish_chinese_variant(&body.text).to_string()
        } else {
            "en".to_string()
        }
    } else {
        source_lang
    };
    let mut dest_index = 0;
    if detected_source == target_lang && body.destination.len() > 1 {
        let fallback_name = &body.destination[1];
        if !fallback_name.trim().is_empty() {
            let alt = hcfy_to_bcp47
                .get(fallback_name.as_str())
                .map(|&s| s.to_string())
                .unwrap_or_else(|| normalize_language_code(fallback_name));
            if !alt.is_empty() {
                dest_index = 1;
                target_lang = alt;
            }
        }
    }

    let paragraphs: Vec<String> = body
        .text
        .split('\n')
        .map(|p| {
            let line = p.trim_end_matches('\r');
            if line.trim().is_empty() {
                String::new()
            } else {
                line.to_string()
            }
        })
        .collect();
    validate_batch(&paragraphs)?;

    let raw_results = service::translate_batch_with_pivot_formats(
        &detected_source,
        &target_lang,
        &paragraphs,
        None,
    )
    .await?;

    let results: Vec<String> = raw_results.into_iter().map(|(res, _)| res).collect();
    Ok(Json(HcfyResponse {
        text: body.text,
        from: bcp47_to_hcfy
            .get(detected_source.as_str())
            .copied()
            .unwrap_or(&detected_source)
            .to_string(),
        to: std::mem::take(&mut body.destination[dest_index]),
        result: Some(results),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcfy_mappings_and_normalization() {
        assert_eq!(HCFY_TO_BCP47.get("越南语"), Some(&"vi"));
        assert_eq!(HCFY_TO_BCP47.get("阿拉伯语"), Some(&"ar"));
        assert_eq!(BCP47_TO_HCFY.get("vi"), Some(&"越南语"));

        // Fallback normalization
        let unmapped = "zh-CN";
        let normalized = HCFY_TO_BCP47
            .get(unmapped)
            .map(|&s| s.to_string())
            .unwrap_or_else(|| normalize_language_code(unmapped));
        assert_eq!(normalized, "zh-Hans");
    }
    #[test]
    fn test_japanese_kana_detection_priority() {
        let text = "日本語です";
        assert!(contains_japanese(text));
        assert!(contains_chinese(text)); // has Kanji
        // Priority must detect Japanese first
        let detected = if contains_japanese(text) {
            "ja"
        } else if contains_chinese(text) {
            "zh"
        } else {
            "en"
        };
        assert_eq!(detected, "ja");
    }
    #[tokio::test]
    async fn test_hcfy_paragraph_batch_limit() {
        let overflow_text = "line\n".repeat(600);
        let req = HcfyRequest {
            name: None,
            text: overflow_text,
            source: Some("英语".to_string()),
            destination: vec!["英语".to_string()],
        };
        let err = translate(Json(req)).await.unwrap_err();
        match err {
            ApiError::BadRequest(msg) => {
                assert!(msg.contains("Batch too large: 601 items (max 512)"), "Unexpected message: {msg}");
            }
            other => panic!("Expected BadRequest, got {other:?}"),
        }

        let valid_req = HcfyRequest {
            name: None,
            text: "hello\nworld".to_string(),
            source: Some("英语".to_string()),
            destination: vec!["英语".to_string()],
        };
        let res = translate(Json(valid_req)).await.unwrap();
        assert_eq!(res.result, Some(vec!["hello".to_string(), "world".to_string()]));
    }
}
