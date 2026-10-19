use mtranserver_lib::api::plugins::deeplx::DeeplxRequest;
use mtranserver_lib::api::plugins::google::GoogleRequest;
use mtranserver_lib::api::plugins::OneOrMany;
use mtranserver_test::loader::load_json;

#[test]
fn test_deeplx_request_fixture() {
    let req: DeeplxRequest = load_json("api/deeplx_request.json");
    assert_eq!(req.text, "Hello world");
    assert_eq!(req.source_lang, "");
    assert_eq!(req.target_lang, "ZH");
}

#[test]
fn test_google_request_fixture() {
    let req: GoogleRequest = load_json("api/google_request.json");
    assert_eq!(req.source, "");
    assert_eq!(req.target, "zh");
    assert_eq!(
        req.q,
        OneOrMany::Many(vec!["Hello".to_string(), "world".to_string()])
    );
}
