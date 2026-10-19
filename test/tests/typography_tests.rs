use mtranserver_lib::engine::text::{
    normalize_academic_and_special_symbols, postprocess_translation_by_language, preprocess_symbols,
};
use mtranserver_test::loader::load_json;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TypographyCase {
    input: String,
    expected: String,
}

#[derive(Debug, Deserialize)]
struct PostprocessCase {
    src: String,
    raw: String,
    src_lang: String,
    trg_lang: String,
    expected: String,
}

#[derive(Debug, Deserialize)]
struct AcademicSymbolCase {
    name: String,
    input: String,
    #[serde(default)]
    expected: Option<String>,
    #[serde(default)]
    expected_contains: Vec<String>,
    #[serde(default)]
    forbidden_contains: Vec<String>,
}

#[test]
fn test_preprocess_symbols_fixtures() {
    let cases: Vec<TypographyCase> = load_json("typography/symbols.json");
    for (idx, case) in cases.iter().enumerate() {
        let processed = preprocess_symbols(&case.input);
        assert_eq!(
            processed.as_ref(),
            case.expected,
            "[typography/symbols.json] case #{} failed for input {:?}",
            idx,
            case.input
        );
    }
}

#[test]
fn test_postprocess_translation_fixtures() {
    let cases: Vec<PostprocessCase> = load_json("typography/postprocess.json");
    for (idx, case) in cases.iter().enumerate() {
        let result = postprocess_translation_by_language(
            &case.raw,
            &case.src,
            &case.src_lang,
            &case.trg_lang,
        );
        assert_eq!(
            result.as_ref(),
            case.expected,
            "[typography/postprocess.json] case #{} failed for raw {:?}",
            idx,
            case.raw
        );
    }
}

#[test]
fn test_academic_symbols_fixtures() {
    let cases: Vec<AcademicSymbolCase> = load_json("typography/academic_symbols.json");
    for (idx, case) in cases.iter().enumerate() {
        let normalized = normalize_academic_and_special_symbols(&case.input);
        if let Some(exp) = &case.expected {
            assert_eq!(
                normalized.as_ref(),
                exp,
                "[typography/academic_symbols.json] case #{} ({}) exact match failed",
                idx,
                case.name
            );
        }
        for item in &case.expected_contains {
            assert!(
                normalized.contains(item),
                "[typography/academic_symbols.json] case #{} ({}) expected to contain {:?}",
                idx,
                case.name,
                item
            );
        }
        for item in &case.forbidden_contains {
            assert!(
                !normalized.contains(item),
                "[typography/academic_symbols.json] case #{} ({}) forbidden to contain {:?}",
                idx,
                case.name,
                item
            );
        }
    }
}
