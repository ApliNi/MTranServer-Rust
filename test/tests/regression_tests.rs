use std::collections::BTreeMap;

use mtranserver_test::loader::load_regression_cases;
use mtranserver_test::regression::{
    default_corpus_path, detect_case_language, load_cases, save_cases, update_cases_file, TestCase,
    COMMON_LANGUAGES,
};

fn test_temp_file(prefix: &str) -> std::path::PathBuf {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_dir = if manifest_dir.ends_with("test") {
        manifest_dir.parent().unwrap().to_path_buf()
    } else {
        manifest_dir
    };
    let dir = root_dir.join(".pi-temp");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!(
        "{}_{}_{}.yml",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn test_regression_cases_format_and_loading() {
    let cases = load_regression_cases();
    assert!(!cases.is_empty(), "Regression cases should not be empty");
    assert!(
        cases.len() >= 60,
        "Expected at least 60 comprehensive test cases, found {}",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        assert!(
            !case.lang.trim().is_empty(),
            "Case #{} has empty lang",
            i + 1
        );
        assert!(
            !case.detected.trim().is_empty(),
            "Case #{} has empty detected language",
            i + 1
        );
        assert!(
            !case.input.trim().is_empty(),
            "Case #{} has empty input",
            i + 1
        );
        assert!(
            !case.output.is_empty(),
            "Case #{} has empty output map",
            i + 1
        );

        for (target, translated) in &case.output {
            assert!(
                !target.trim().is_empty(),
                "Case #{} has empty output target language",
                i + 1
            );
            assert!(
                !translated.trim().is_empty(),
                "Case #{} has empty output text for target '{}'",
                i + 1,
                target
            );
            assert_ne!(
                target,
                &case.lang,
                "Case #{} output should exclude source lang '{}'",
                i + 1,
                case.lang
            );
            assert_ne!(
                target,
                &case.detected,
                "Case #{} output should exclude detected language '{}'",
                i + 1,
                case.detected
            );
        }
    }
}

#[tokio::test]
async fn test_update_cases_mock() {
    let temp_file = test_temp_file("regression_update_test");

    let initial_cases = vec![
        TestCase {
            lang: "en".into(),
            detected: String::new(),
            input: "Hello world".into(),
            output: BTreeMap::new(),
        },
        TestCase {
            lang: "en".into(),
            detected: String::new(),
            input: "Thank you".into(),
            output: BTreeMap::new(),
        },
    ];

    save_cases(&temp_file, &initial_cases).expect("Failed to save initial cases");

    let count = update_cases_file(&temp_file, true)
        .await
        .expect("Failed to update cases file");
    assert_eq!(count, 2);

    let updated_cases = load_cases(&temp_file).expect("Failed to load updated cases");
    assert_eq!(updated_cases.len(), 2);

    // Case 0: "Hello world"
    assert_eq!(updated_cases[0].lang, "en");
    assert_eq!(updated_cases[0].detected, "en");
    assert!(!updated_cases[0].output.contains_key("en"));
    assert_eq!(
        updated_cases[0].output.get("zh-Hans").map(String::as_str),
        Some("[zh-Hans:Hello world]")
    );
    assert_eq!(
        updated_cases[0].output.get("ja").map(String::as_str),
        Some("[ja:Hello world]")
    );
    for &target in COMMON_LANGUAGES {
        if target == "en" {
            assert!(!updated_cases[0].output.contains_key(target));
        } else {
            let expected = format!("[{}:Hello world]", target);
            assert_eq!(
                updated_cases[0].output.get(target),
                Some(&expected),
                "Missing or mismatched mock translation for target '{}'",
                target
            );
        }
    }

    // Case 1: "Thank you"
    assert_eq!(updated_cases[1].lang, "en");
    assert_eq!(updated_cases[1].detected, "en");
    assert!(!updated_cases[1].output.contains_key("en"));
    assert_eq!(
        updated_cases[1].output.get("zh-Hans").map(String::as_str),
        Some("[zh-Hans:Thank you]")
    );
    assert_eq!(
        updated_cases[1].output.get("ja").map(String::as_str),
        Some("[ja:Thank you]")
    );
    for &target in COMMON_LANGUAGES {
        if target == "en" {
            assert!(!updated_cases[1].output.contains_key(target));
        } else {
            let expected = format!("[{}:Thank you]", target);
            assert_eq!(
                updated_cases[1].output.get(target),
                Some(&expected),
                "Missing or mismatched mock translation for target '{}'",
                target
            );
        }
    }

    let _ = std::fs::remove_file(&temp_file);
}

#[tokio::test]
async fn test_detect_case_language() {
    assert_eq!(detect_case_language("Hello world").await, "en");
    assert_eq!(detect_case_language("你好，世界").await, "zh-Hans");
    assert_eq!(detect_case_language("Bonjour tout le monde").await, "fr");
}

#[tokio::test]
#[ignore = "Requires local Marian NMT models in data/models/"]
async fn test_real_service_regression_update() {
    use mtranserver_test::regression::init_test_environment;

    init_test_environment()
        .await
        .expect("Failed to init service");
    let corpus_path = default_corpus_path();

    println!(
        "Updating regression translation outputs in {:?}...",
        corpus_path
    );
    let count = update_cases_file(&corpus_path, false)
        .await
        .expect("Failed to update cases");
    println!(
        "Successfully updated {} cases in {:?}. Inspect differences with `git diff`.",
        count, corpus_path
    );
}
