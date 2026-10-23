use mtranserver_test::loader::load_regression_cases;
use mtranserver_test::regression::{
    default_corpus_path, load_cases, save_cases, update_cases_file, TestCase,
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
        "{}_{}_{}.json",
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
            !case.target.trim().is_empty(),
            "Case #{} has empty target lang",
            i + 1
        );
        assert!(
            !case.input.trim().is_empty(),
            "Case #{} has empty input",
            i + 1
        );
        assert!(
            !case.output.trim().is_empty(),
            "Case #{} has empty output",
            i + 1
        );
    }
}

#[tokio::test]
async fn test_update_cases_mock() {
    let temp_file = test_temp_file("regression_update_test");

    let initial_cases = vec![
        TestCase {
            target: "zh-Hans".into(),
            input: "Hello world".into(),
            output: "Old outdated translation".into(),
        },
        TestCase {
            target: "ja".into(),
            input: "Thank you".into(),
            output: "".into(),
        },
    ];

    save_cases(&temp_file, &initial_cases).expect("Failed to save initial cases");

    let count = update_cases_file(&temp_file, true)
        .await
        .expect("Failed to update cases file");
    assert_eq!(count, 2);

    let updated_cases = load_cases(&temp_file).expect("Failed to load updated cases");
    assert_eq!(updated_cases[0].output, "[zh-Hans:Hello world]");
    assert_eq!(updated_cases[1].output, "[ja:Thank you]");

    let _ = std::fs::remove_file(&temp_file);
}

#[tokio::test]
#[ignore = "Requires local Marian NMT models in data/models/"]
async fn test_real_service_regression_update() {
    use mtranserver_test::regression::init_test_environment;

    init_test_environment()
        .await
        .expect("Failed to init service");
    let corpus_path = default_corpus_path();

    println!("Updating regression translation outputs in {:?}...", corpus_path);
    let count = update_cases_file(&corpus_path, false)
        .await
        .expect("Failed to update cases");
    println!(
        "Successfully updated {} cases in {:?}. Inspect differences with `git diff`.",
        count, corpus_path
    );
}
