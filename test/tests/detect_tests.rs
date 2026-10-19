use mtranserver_lib::engine::rules::detect::{detect_dominant, detect_text_language};
use mtranserver_test::loader::load_json;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DetectCase {
    #[serde(default)]
    lang: Option<String>,
    input: String,
    expected: String,
}

fn run_detect_cases(path: &str) {
    let cases: Vec<DetectCase> = load_json(path);
    for (idx, case) in cases.iter().enumerate() {
        let detected = detect_text_language(&case.input);
        assert_eq!(
            detected, case.expected,
            "[{}] failed at case #{} ({:?}) for input {:?}: expected {}, got {}",
            path, idx, case.lang, case.input, case.expected, detected
        );
    }
}

#[test]
fn test_detect_tier0_fast_bypass() {
    run_detect_cases("detect/fast_bypass.json");
}

#[test]
fn test_detect_tier1_and_tier2_unique_scripts() {
    run_detect_cases("detect/unique_scripts.json");
}

#[test]
fn test_detect_tier3_cjk_and_kanji() {
    run_detect_cases("detect/cjk_and_kanji.json");
}

#[test]
fn test_detect_tier4_cyrillic_and_indic() {
    run_detect_cases("detect/cyrillic_and_indic.json");
}

#[test]
fn test_detect_tier4_latin_orthography() {
    run_detect_cases("detect/latin_orthography.json");
}

#[test]
fn test_detect_tier3_core_greetings() {
    run_detect_cases("detect/greetings.json");
}

#[test]
fn test_detect_global_57_languages() {
    run_detect_cases("detect/global_57_languages.json");
}

#[test]
fn test_detect_dominant_mapping() {
    let cases: Vec<DetectCase> = load_json("detect/dominant.json");
    for (idx, case) in cases.iter().enumerate() {
        let dominant = detect_dominant(&case.input);
        assert_eq!(
            dominant, case.expected,
            "[detect/dominant.json] failed at case #{}: expected {}, got {}",
            idx, case.expected, dominant
        );
    }
}
