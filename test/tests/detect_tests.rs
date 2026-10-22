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

#[test]
fn test_tier0_does_not_falsely_bypass_natural_language() {
    // Spanish: "cargo" must not be bypassed by Tier 0
    let s1 = "Este es un cargo muy importante en la empresa pública española.";
    assert_eq!(detect_text_language(s1), "es");

    // Turkish: "git" must not be bypassed by Tier 0
    let s2 = "Buradan hemen git ve bir daha asla buraya geri dönme.";
    assert_eq!(detect_text_language(s2), "tr");

    // Norwegian: "tar" must not be bypassed by Tier 0
    let s3 = "Hver morgen tar jeg bussen til skolen for å møte vennene mine i Oslo.";
    assert_eq!(detect_text_language(s3), "nb");
    // Spanish: "sudo" must not be bypassed by Tier 0
    let s4 = "Siempre sudo mucho cuando salgo a correr por la tarde en el parque.";
    assert_eq!(detect_text_language(s4), "es");

    // Danish: "let", "var" must not be bypassed by Tier 0
    let s5 = "Dette er en let opgave som alle børn i skolen kan løse uden problemer.";
    assert_eq!(detect_text_language(s5), "da");

    let s6 = "Hvor var du henne i går aftes da vi spiste aftensmad sammen?";
    assert_eq!(detect_text_language(s6), "da");

    // French: "import" must not be bypassed by Tier 0
    let s7 = "Notre entreprise est spécialisée dans l'import de vins et fromages français.";
    assert_eq!(detect_text_language(s7), "fr");

    // True CLI / Code / Greeting patterns must still be recognized by Tier 0
    assert_eq!(detect_text_language("cargo build --release"), "en");
    assert_eq!(detect_text_language("git checkout -b feature/auth"), "en");
    assert_eq!(detect_text_language("tar -czvf archive.tar.gz file.txt"), "en");
    assert_eq!(detect_text_language("sudo apt update && sudo apt upgrade"), "en");
    assert_eq!(detect_text_language("let count: usize = 42;"), "en");
    assert_eq!(detect_text_language("var active = true;"), "en");
    assert_eq!(detect_text_language("import React from 'react';"), "en");
    assert_eq!(detect_text_language("SELECT * FROM users WHERE id = 42;"), "en");
    assert_eq!(detect_text_language("Hello world!"), "en");
}
