use mtranserver_lib::engine::markup::split::{reassemble_sentences, split_sentences, split_sentences_with_lang};
use mtranserver_test::loader::load_json;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SentencePart {
    text: String,
    separator: String,
}

#[derive(Debug, Deserialize)]
struct SplitTestCase {
    name: String,
    #[serde(default)]
    lang: Option<String>,
    text: String,
    expected: Vec<SentencePart>,
}

fn run_split_cases(path: &str) {
    let cases: Vec<SplitTestCase> = load_json(path);
    for (idx, case) in cases.iter().enumerate() {
        let sentences = match &case.lang {
            Some(l) => split_sentences_with_lang(&case.text, Some(l.as_str())),
            None => split_sentences(&case.text),
        };

        assert_eq!(
            sentences.len(),
            case.expected.len(),
            "[{}] case #{} ({}) sentence count mismatch: expected {}, got {}",
            path,
            idx,
            case.name,
            case.expected.len(),
            sentences.len()
        );

        for (s_idx, (actual, exp)) in sentences.iter().zip(&case.expected).enumerate() {
            assert_eq!(
                actual.text, exp.text,
                "[{}] case #{} ({}) part #{} text mismatch",
                path, idx, case.name, s_idx
            );
            assert_eq!(
                actual.separator, exp.separator,
                "[{}] case #{} ({}) part #{} separator mismatch",
                path, idx, case.name, s_idx
            );
        }

        assert_eq!(
            reassemble_sentences(&sentences),
            case.text,
            "[{}] case #{} ({}) reassembly mismatch",
            path,
            idx,
            case.name
        );
    }
}

#[test]
fn test_sentence_splitting_fixtures() {
    run_split_cases("split/sentences.json");
}

#[test]
fn test_multilingual_sentence_splitting_fixtures() {
    run_split_cases("split/multilingual_sentences.json");
}
