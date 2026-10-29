use mtranserver_lib::engine::text::{
    normalize_academic_and_special_symbols, preprocess_symbols,
};
use mtranserver_test::loader::load_json;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TypographyCase {
    input: String,
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
#[test]
fn test_preserve_formulas_academic_symbols() {
    assert_eq!(normalize_academic_and_special_symbols("E=mc²").as_ref(), "E=mc²");
    assert_eq!(normalize_academic_and_special_symbols("H₂O").as_ref(), "H₂O");
}

use mtranserver_lib::service::{
    apply_quote_style, apply_truecase_western, format_french_spacing, format_pangu_spacing,
    format_spanish_inverted, format_zh_punctuation, normalize_output_punctuation,
    postprocess_typography, truecase_western, TranslatedText,
};
use std::borrow::Cow;

#[test]
fn test_normalize_output_punctuation() {
    let s = "你好, 世界!".to_string();
    let norm = normalize_output_punctuation(s, "zh");
    assert_eq!(norm.as_str(), "你好，世界！");

    let s2 = "hello, world".to_string();
    let norm2 = normalize_output_punctuation(s2, "en");
    assert_eq!(norm2.as_str(), "Hello, world");
}

#[test]
fn test_format_zh_punctuation_protections() {
    let num1 = "总金额为 1,000,000 美元";
    assert_eq!(format_zh_punctuation(num1, "zh"), "总金额为 1,000,000 美元");
    assert!(matches!(format_zh_punctuation(num1, "zh"), Cow::Borrowed(_)));

    let num2 = "欧洲格式 3.141,59 测试";
    assert_eq!(format_zh_punctuation(num2, "zh"), "欧洲格式 3.141,59 测试");

    let list = "第 1, 2, 3 项";
    assert_eq!(format_zh_punctuation(list, "zh"), "第 1，2，3 项");

    let time = "会议时间 12:30 开始";
    assert_eq!(format_zh_punctuation(time, "zh"), "会议时间 12:30 开始");

    let url = "访问 https://example.com/api";
    assert_eq!(format_zh_punctuation(url, "zh"), "访问 https://example.com/api");

    let host1 = "服务运行在 localhost:8080 地址";
    assert_eq!(format_zh_punctuation(host1, "zh"), "服务运行在 localhost:8080 地址");
    let host2 = "连接 127.0.0.1:3000 端口";
    assert_eq!(format_zh_punctuation(host2, "zh"), "连接 127.0.0.1:3000 端口");

    let rust_path = "使用 std::path::PathBuf 结构体";
    assert_eq!(format_zh_punctuation(rust_path, "zh"), "使用 std::path::PathBuf 结构体");
    assert!(matches!(format_zh_punctuation(rust_path, "zh"), Cow::Borrowed(_)));

    let ipv6 = "连接 [2001:db8::1]:8080 节点";
    assert_eq!(format_zh_punctuation(ipv6, "zh"), "连接 [2001:db8::1]:8080 节点");

    let header = "Content-Type: application/json\nHost: 127.0.0.1";
    assert_eq!(format_zh_punctuation(header, "zh"), "Content-Type: application/json\nHost: 127.0.0.1");

    let url_params = "详情参见 https://example.com/api?a=1,b=2;c=3 页面, 完成!";
    assert_eq!(
        format_zh_punctuation(url_params, "zh"),
        "详情参见 https://example.com/api?a=1,b=2;c=3 页面，完成！"
    );
    let label = "注意: 请勿操作";
    assert_eq!(format_zh_punctuation(label, "zh"), "注意：请勿操作");

    let md_inline = "执行 `let a = 1, b: 2;` 命令, 完成!";
    assert_eq!(
        format_zh_punctuation(md_inline, "zh"),
        "执行 `let a = 1, b: 2;` 命令，完成！"
    );

    let md_block = "如下: ```std::path, 1,000,000!``` 结束";
    assert_eq!(
        format_zh_punctuation(md_block, "zh"),
        "如下：```std::path, 1,000,000!``` 结束"
    );

    assert_eq!(
        format_zh_punctuation("如果 value != 0, 则继续执行", "zh"),
        "如果 value != 0，则继续执行"
    );
    assert_eq!(
        format_zh_punctuation("使用 obj?.property 访问可选属性", "zh"),
        "使用 obj?.property 访问可选属性"
    );
    assert_eq!(
        format_zh_punctuation("设置 flag = !isActive 来取反", "zh"),
        "设置 flag = !isActive 来取反"
    );
    assert_eq!(
        format_zh_punctuation("使用 value ?? default 作为备选", "zh"),
        "使用 value ?? default 作为备选"
    );

    let json_input = r#"配置 {"host": "localhost"} 成功!"#;
    assert_eq!(
        format_zh_punctuation(json_input, "zh"),
        "配置 {\"host\": \"localhost\"} 成功！"
    );
    let macro_input = "执行 println!(\"ok\") 和 assert!(true) 测试完成!";
    assert_eq!(
        format_zh_punctuation(macro_input, "zh"),
        "执行 println!(\"ok\") 和 assert!(true) 测试完成！"
    );

    let shell_input = "配置模板:将 PORT=${HTTP_PORT:-8080} 导出,并使用 max_retries=%d 设置 SECRET=\"{{APP_SECRET_KEY}}\"。";
    let shell_res = format_zh_punctuation(shell_input, "zh");
    assert_eq!(
        shell_res,
        "配置模板：将 PORT=${HTTP_PORT:-8080} 导出，并使用 max_retries=%d 设置 SECRET=\"{{APP_SECRET_KEY}}\"。"
    );

    let dropped_quote = "配置模板:将 PORT=${HTTP_PORT:-8080} 导出,并使用 max_retries=%d 设置 SECRET=\"{APP_SECRET_KEY}}。";
    let dropped_res = postprocess_typography(Cow::Borrowed(dropped_quote), "zh-Hans");
    assert_eq!(
        dropped_res.as_ref(),
        "配置模板：将 PORT=${HTTP_PORT:-8080} 导出，并使用 max_retries=%d 设置 SECRET=\"{APP_SECRET_KEY}}。"
    );

    assert_eq!(
        format_zh_punctuation("推荐使用 Docker: 容器平台", "zh"),
        "推荐使用 Docker：容器平台"
    );
    assert_eq!(
        format_zh_punctuation("参考 API: 接口定义", "zh"),
        "参考 API：接口定义"
    );
    assert_eq!(
        format_zh_punctuation("步骤 1: 点击确认", "zh"),
        "步骤 1：点击确认"
    );
}

#[test]
fn test_truecase_western_cjk_exemption() {
    let input_ja = "iphone は非常に便利です。";
    let out_ja = apply_truecase_western(Cow::Borrowed(input_ja), "ja");
    assert_eq!(out_ja.as_ref(), "iphone は非常に便利です。");

    let input_ko = "ebay 에서 구입했습니다.";
    let out_ko = apply_truecase_western(Cow::Borrowed(input_ko), "ko");
    assert_eq!(out_ko.as_ref(), "ebay 에서 구입했습니다.");

    let input_en = "iphone is very convenient.";
    let out_en = apply_truecase_western(Cow::Borrowed(input_en), "en");
    assert_eq!(out_en.as_ref(), "Iphone is very convenient.");
}

#[test]
fn test_format_zh_math_comparison_not_html() {
    let input = "数量 < 100MB, 且折扣 > 5折!";
    let formatted = format_zh_punctuation(input, "zh");
    assert_eq!(formatted, "数量 < 100 MB，且折扣 > 5 折！");
}

#[test]
fn test_format_zh_ipv6_colons_preserved() {
    let input = "服务地址: 2001:db8::1, 端口: 8080";
    let formatted = format_zh_punctuation(input, "zh");
    assert_eq!(formatted, "服务地址：2001:db8::1，端口：8080");
}

#[test]
fn test_format_zh_pangu_cjk_extended_zero() {
    let input = "二〇二六年5月";
    let formatted = format_zh_punctuation(input, "zh");
    assert_eq!(formatted, "二〇二六年 5 月");

    let input_repeat = "常々100次";
    let formatted_repeat = format_zh_punctuation(input_repeat, "zh");
    assert_eq!(formatted_repeat, "常々 100 次");
}

#[test]
fn test_truecase_western_accents_and_punctuation() {
    assert_eq!(truecase_western("été chaud. über alles.", "fr"), "Été chaud. Über alles.");
    assert_eq!(truecase_western("¿cómo estás? ¡muy bien!", "es"), "¿Cómo estás? ¡Muy bien!");
    assert_eq!(truecase_western("«bonjour monde»", "fr"), "«Bonjour monde»");
    assert_eq!(truecase_western("— yes, indeed.", "en"), "— Yes, indeed.");
    assert_eq!(
        truecase_western("please contact us vs. them immediately.", "en"),
        "Please contact us vs. them immediately."
    );
    assert_eq!(
        truecase_western("the box measures 10 in. wide by 5 in. deep.", "en"),
        "The box measures 10 in. wide by 5 in. deep."
    );
    assert_eq!(
        truecase_western("use abbreviations e.g. for example or i.e. that is.", "en"),
        "Use abbreviations e.g. for example or i.e. that is."
    );
    assert_eq!(
        truecase_western("she bought apples, etc. from the store.", "en"),
        "She bought apples, etc. from the store."
    );
    assert_eq!(
        truecase_western("пожалуйста, см. ниже подробности.", "ru"),
        "Пожалуйста, см. ниже подробности."
    );
    assert_eq!(
        truecase_western("ein beispiel, z. b. dieses hier.", "de"),
        "Ein beispiel, z. b. dieses hier."
    );
}

#[test]
fn test_format_zh_pangu_spacing() {
    assert_eq!(format_zh_punctuation("使用Rust开发", "zh"), "使用 Rust 开发");
    assert_eq!(format_zh_punctuation("在Rust中", "zh"), "在 Rust 中");
    assert_eq!(format_zh_punctuation("延迟为10ms", "zh"), "延迟为 10ms");
    assert_eq!(format_zh_punctuation("100MB内存", "zh"), "100 MB 内存");
    assert_eq!(format_zh_punctuation("100MB 内存", "zh"), "100 MB 内存");
    assert_eq!(format_zh_punctuation("100 MB内存", "zh"), "100 MB 内存");

    assert_eq!(format_pangu_spacing("在Rust中"), "在 Rust 中");

    let spaced = "在 Rust 中使用 100 MB 内存";
    assert_eq!(format_zh_punctuation(spaced, "zh"), spaced);
    assert!(matches!(format_zh_punctuation(spaced, "zh"), Cow::Borrowed(_)));

    assert_eq!(format_zh_punctuation("使用Rust，开发高效", "zh"), "使用 Rust，开发高效");
    assert_eq!(format_zh_punctuation("你好！World", "zh"), "你好！World");
    assert_eq!(format_zh_punctuation("“Rust”权威指南", "zh"), "“Rust”权威指南");
    assert_eq!(format_zh_punctuation("性能提升了（约10%）", "zh"), "性能提升了（约 10%）");
}

#[test]
fn test_format_zh_markup_and_url_protections() {
    let html1 = "<span class=\"cjk\">Rust开发</span>";
    assert_eq!(format_zh_punctuation(html1, "zh"), "<span class=\"cjk\">Rust 开发</span>");
    let html2 = "<input value=\"在Rust中\" />";
    assert_eq!(format_zh_punctuation(html2, "zh"), "<input value=\"在Rust中\" />");
    let html3 = "<div>在Rust中</div>";
    assert_eq!(format_zh_punctuation(html3, "zh"), "<div>在 Rust 中</div>");

    let md1 = "执行 `let a = 1;` 命令";
    assert_eq!(format_zh_punctuation(md1, "zh"), "执行 `let a = 1;` 命令");
    let md2 = "运行 `cargo build 100MB` 工具";
    assert_eq!(format_zh_punctuation(md2, "zh"), "运行 `cargo build 100MB` 工具");
    let md3 = "在`cargo test`提交";
    assert_eq!(format_zh_punctuation(md3, "zh"), "在 `cargo test` 提交");
    assert_eq!(format_pangu_spacing(md3), "在 `cargo test` 提交");

    let md4 = "运行`cargo run`。";
    assert_eq!(format_zh_punctuation(md4, "zh"), "运行 `cargo run`。");

    let md5 = "`cargo test`是测试工具";
    assert_eq!(format_zh_punctuation(md5, "zh"), "`cargo test` 是测试工具");

    let url1 = "访问 https://example.com/api100MB 测试";
    assert_eq!(format_zh_punctuation(url1, "zh"), "访问 https://example.com/api100MB 测试");
    let url2 = "访问https://example.com/api测试";
    assert_eq!(format_zh_punctuation(url2, "zh"), "访问 https://example.com/api 测试");
}

#[test]
fn test_western_truecasing() {
    assert_eq!(truecase_western("hello world", "en"), "Hello world");
    assert_eq!(
        truecase_western("hello world. this is rust! how are you? good.", "en"),
        "Hello world. This is rust! How are you? Good."
    );
    assert_eq!(truecase_western("version 3.14 is released.", "en"), "Version 3.14 is released.");
    assert_eq!(truecase_western("visit https://example.com/api now.", "en"), "Visit https://example.com/api now.");
    assert_eq!(truecase_western("<p>hello world</p>", "en"), "<p>Hello world</p>");
    assert_eq!(truecase_western("`code` is untouched.", "en"), "`code` is untouched.");

    let cap = "Hello world. This is capitalized.";
    assert!(matches!(truecase_western(cap, "en"), Cow::Borrowed(_)));

    assert_eq!(truecase_western("bonjour le monde", "fr"), "Bonjour le monde");
    assert_eq!(truecase_western("hello world", "zh"), "hello world");
    assert!(matches!(truecase_western("hello world", "zh"), Cow::Borrowed(_)));
}

#[test]
fn test_defensive_char_boundary_slicing() {
    let s = "这是一段测试文字100MB";
    let spaced = format_pangu_spacing(s);
    assert_eq!(spaced, "这是一段测试文字 100 MB");

    let tc = truecase_western("hello world", "en");
    assert_eq!(tc, "Hello world");
}

#[test]
fn test_normalize_punctuation_decoupled_truecasing() {
    let text_en = TranslatedText::String("hello world. good morning.".to_string());
    let normalized_en = text_en.normalize_punctuation("en");
    assert_eq!(normalized_en.as_str(), "Hello world. Good morning.");

    let text_zh_on = TranslatedText::String("你好,世界!".to_string());
    let converted_zh = text_zh_on.normalize_punctuation("zh");
    assert_eq!(converted_zh.as_str(), "你好，世界！");
}

#[test]
fn test_format_french_spacing() {
    let text = "Comment allez-vous? Bien! Merci; et vous?";
    let formatted = format_french_spacing(Cow::Borrowed(text));
    assert_eq!(formatted, "Comment allez-vous\u{202F}? Bien\u{202F}! Merci\u{202F}; et vous\u{202F}?");

    let colon_text = "Attention : ceci est important.";
    let formatted_colon = format_french_spacing(Cow::Borrowed(colon_text));
    assert_eq!(formatted_colon, "Attention\u{00A0}: ceci est important.");

    let no_space = "Exemple:voici";
    let formatted_ns = format_french_spacing(Cow::Borrowed(no_space));
    assert_eq!(formatted_ns, "Exemple\u{00A0}:voici");

    let plain = "Bonjour le monde.";
    let res = format_french_spacing(Cow::Borrowed(plain));
    assert!(matches!(res, Cow::Borrowed(_)));

    let url = "Visitez https://example.com/test?q=1 pour voir!";
    let formatted_url = format_french_spacing(Cow::Borrowed(url));
    assert_eq!(formatted_url, "Visitez https://example.com/test?q=1 pour voir\u{202F}!");

    let code = "Code `let x: i32 = 1;` fonctionne!";
    let formatted_code = format_french_spacing(Cow::Borrowed(code));
    assert_eq!(formatted_code, "Code `let x: i32 = 1;` fonctionne\u{202F}!");

    let html = "<span>Texte?</span>";
    let formatted_html = format_french_spacing(Cow::Borrowed(html));
    assert_eq!(formatted_html, "<span>Texte\u{202F}?</span>");

    let ipv6_fr = "Serveur [2001:db8::1]:8080 actif";
    let formatted_ipv6 = format_french_spacing(Cow::Borrowed(ipv6_fr));
    assert_eq!(formatted_ipv6, "Serveur [2001:db8::1]:8080 actif");

    let header_fr = "Content-Type: text/html";
    let formatted_header = format_french_spacing(Cow::Borrowed(header_fr));
    assert_eq!(formatted_header, "Content-Type: text/html");

    let guillemets_unspaced = "Citation «Texte» et «Autre»";
    let formatted_guill1 = format_french_spacing(Cow::Borrowed(guillemets_unspaced));
    assert_eq!(formatted_guill1, "Citation «\u{202F}Texte\u{202F}» et «\u{202F}Autre\u{202F}»");

    let guillemets_spaced = "Citation « Texte » et « Autre »";
    let formatted_guill2 = format_french_spacing(Cow::Borrowed(guillemets_spaced));
    assert_eq!(formatted_guill2, "Citation «\u{202F}Texte\u{202F}» et «\u{202F}Autre\u{202F}»");

    let empty_guill = "Guillemets vides: «»";
    let formatted_empty = format_french_spacing(Cow::Borrowed(empty_guill));
    assert_eq!(formatted_empty, "Guillemets vides\u{00A0}: «»");

    let opt_chain = "valeur = user?.name ? 'oui' : 'non';";
    let formatted_opt = format_french_spacing(Cow::Borrowed(opt_chain));
    assert_eq!(formatted_opt, "valeur = user?.name\u{202F}? 'oui'\u{00A0}: 'non';");

    let assert_dot = "let x = node!.value;";
    let formatted_assert = format_french_spacing(Cow::Borrowed(assert_dot));
    assert_eq!(formatted_assert, "let x = node!.value\u{202F};");

    let method_dot = "Appelez record.save! pour persister.";
    let formatted_mdot = format_french_spacing(Cow::Borrowed(method_dot));
    assert_eq!(formatted_mdot, "Appelez record.save! pour persister.");

    let method_inline = "Utilisez save! pour sauvegarder.";
    let formatted_minline = format_french_spacing(Cow::Borrowed(method_inline));
    assert_eq!(formatted_minline, "Utilisez save! pour sauvegarder.");

    let macro_call = "println!(\"message\");";
    let formatted_macro = format_french_spacing(Cow::Borrowed(macro_call));
    assert_eq!(formatted_macro, "println!(\"message\")\u{202F};");
}

#[test]
fn test_format_spanish_inverted_punctuation() {
    let q = "Cómo estás?";
    let formatted_q = format_spanish_inverted(Cow::Borrowed(q));
    assert_eq!(formatted_q, "¿Cómo estás?");

    let e = "Qué bueno!";
    let formatted_e = format_spanish_inverted(Cow::Borrowed(e));
    assert_eq!(formatted_e, "¡Qué bueno!");

    let already = "¿Cómo estás? ¡Muy bien!";
    let formatted_already = format_spanish_inverted(Cow::Borrowed(already));
    assert!(matches!(formatted_already, Cow::Borrowed(_)));

    let sub_q = "Hola, cómo estás?";
    let formatted_sub = format_spanish_inverted(Cow::Borrowed(sub_q));
    assert_eq!(formatted_sub, "Hola, ¿cómo estás?");

    let sub_intro = "Por favor, dime la verdad!";
    let formatted_intro = format_spanish_inverted(Cow::Borrowed(sub_intro));
    assert_eq!(formatted_intro, "Por favor, ¡dime la verdad!");

    let plain = "Buenos días a todos.";
    let res = format_spanish_inverted(Cow::Borrowed(plain));
    assert!(matches!(res, Cow::Borrowed(_)));

    let comp_q1 = "Hola, por qué no vienes?";
    let formatted_comp1 = format_spanish_inverted(Cow::Borrowed(comp_q1));
    assert_eq!(formatted_comp1, "Hola, ¿por qué no vienes?");

    let comp_q2 = "Dime, para qué sirve esto?";
    let formatted_comp2 = format_spanish_inverted(Cow::Borrowed(comp_q2));
    assert_eq!(formatted_comp2, "Dime, ¿para qué sirve esto?");
}

#[test]
fn test_postprocess_typography_cjk_pangu() {
    let zh_text = "使用Rust开发100MB内存应用";
    let processed_zh = postprocess_typography(Cow::Borrowed(zh_text), "zh");
    assert_eq!(processed_zh, "使用 Rust 开发 100 MB 内存应用");

    let ja_text = "Rustで開発する100MBのメモリ";
    let processed_ja = postprocess_typography(Cow::Borrowed(ja_text), "ja");
    assert_eq!(processed_ja, ja_text);

    let ko_text = "Rust로개발하는100MB메모리";
    let processed_ko = postprocess_typography(Cow::Borrowed(ko_text), "ko");
    assert_eq!(processed_ko, ko_text);
}

#[test]
fn test_postprocess_typography_quote_styles() {
    let fr_text = "Il a dit: \"Bonjour le monde\"!";
    let processed_fr = postprocess_typography(Cow::Borrowed(fr_text), "fr");
    assert_eq!(processed_fr, "Il a dit\u{00A0}: «Bonjour le monde»!");

    let de_text = "Er sagte: \"Hallo Welt\".";
    let processed_de = postprocess_typography(Cow::Borrowed(de_text), "de");
    assert_eq!(processed_de, "Er sagte: „Hallo Welt“.");

    let ja_text = "彼は\"こんにちは世界\"と言った。";
    let processed_ja = postprocess_typography(Cow::Borrowed(ja_text), "ja");
    assert_eq!(processed_ja, "彼は「こんにちは世界」と言った。");

    let code_html_url = "Citation \"Texte\" avec `<tag attr=\"1\">` et \"Autre\"!";
    let processed_mixed = postprocess_typography(Cow::Borrowed(code_html_url), "fr");
    assert_eq!(processed_mixed, "Citation «Texte» avec `<tag attr=\"1\">` et «Autre»!");

    let url_quote = "Visitez https://example.com/api?q=rust pour \"tester\"!";
    let processed_url = postprocess_typography(Cow::Borrowed(url_quote), "fr");
    assert_eq!(processed_url, "Visitez https://example.com/api?q=rust pour «tester»!");

    let en_text = "He said: \"Hello world\"!";
    let processed_en = postprocess_typography(Cow::Borrowed(en_text), "en");
    assert_eq!(processed_en, "He said: “Hello world”!");

    let inch_text = "A 14\" display and 27\" monitor with 3.5\" drive.";
    let processed_inch = apply_quote_style(Cow::Borrowed(inch_text), "curly");
    assert_eq!(processed_inch, "A 14\" display and 27\" monitor with 3.5\" drive.");

    let fastpath_text = "Zero allocation path 14\" screen without convertible quotes";
    match apply_quote_style(Cow::Borrowed(fastpath_text), "curly") {
        Cow::Borrowed(_) => {}
        Cow::Owned(_) => panic!("Expected Cow::Borrowed for zero-allocation fast path"),
    }

    let ctx_text = "He said (\"nested\"), and then \"quoted\"!";
    let processed_ctx = apply_quote_style(Cow::Borrowed(ctx_text), "curly");
    assert_eq!(processed_ctx, "He said (“nested”), and then “quoted”!");
}

#[test]
fn test_postprocess_typography_spanish_inverted_truecasing() {
    let q = "cómo estás?";
    let processed_q = postprocess_typography(Cow::Borrowed(q), "es");
    assert_eq!(processed_q, "¿Cómo estás?");

    let sub_q = "hola, cómo estás?";
    let processed_sub = postprocess_typography(Cow::Borrowed(sub_q), "es");
    assert_eq!(processed_sub, "Hola, ¿Cómo estás?");

    let comp_q = "hola, por qué no vienes?";
    let processed_comp = postprocess_typography(Cow::Borrowed(comp_q), "es");
    assert_eq!(processed_comp, "Hola, ¿Por qué no vienes?");

    let sub_e = "por favor, dime la verdad!";
    let processed_e = postprocess_typography(Cow::Borrowed(sub_e), "es");
    assert_eq!(processed_e, "Por favor, ¡Dime la verdad!");
}
