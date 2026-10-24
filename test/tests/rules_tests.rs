use mtranserver_lib::engine::rules::LanguageRuleProfile;
use mtranserver_test::loader::load_text;

#[test]
fn test_en_sample_rules() {
    let toml_str = load_text("rules/en_sample.toml");
    let profile = LanguageRuleProfile::from_toml_str(&toml_str)
        .expect("Failed to parse en_sample.toml");
    assert_eq!(profile.language, "en");
    assert!(profile.word_boundary);

    // Nonbreaking prefix
    assert!(profile.is_nonbreaking_prefix("Hello Mr.", 8, 6));
    assert!(profile.is_nonbreaking_prefix("See e.g.", 7, 4));
    assert!(!profile.is_nonbreaking_prefix("Hello World.", 11, 6));

    // Clause conjunctions
    assert!(profile.matches_clause_conjunction(" and then"));
    assert!(profile.is_conjunction_clause_marker(" even though it happened"));
    assert!(profile.matches_clause_conjunction(" \"but we know\""));
    assert!(!profile.matches_clause_conjunction(" android is an OS"));

    // Discourse starters
    assert!(profile.is_discourse_chunk_starter("It worked fine."));
    assert!(profile.is_discourse_chunk_starter(" \"This is great.\""));
    assert!(!profile.is_discourse_chunk_starter("Item number 42."));

    // Conditional end prefixes
    assert!(profile.is_conditional_end("etc."));
    assert!(profile.is_conditional_end("ETC."));
    assert!(profile.is_conditional_end("al."));
    assert!(!profile.is_conditional_end("Mr."));

    // Hyphenation rules
    assert!(profile.is_hyphen_protected("pre", "commit"));
    assert!(profile.is_hyphen_protected("gluten", "free"));
    assert!(profile.is_hyphen_protected("word", "word"));
    assert!(profile.is_hyphen_protected("B2B", "services"));
    assert!(!profile.is_hyphen_protected("snap", "compacted"));
    assert!(!profile.is_hyphen_protected("Snap", "Compacted"));

    let reg = mtranserver_lib::engine::rules::RuleRegistry::global();
    assert!(reg.is_hyphen_protected("pre", "commit"));
    assert!(reg.is_hyphen_protected("B2B", "solution"));
    assert!(!reg.is_hyphen_protected("snap", "compacted"));
    assert!(!reg.is_hyphen_protected("Snap", "Compacted"));
    assert!(reg.is_hyphen_protected_lang("en", "gluten", "free"));
}

#[test]
fn test_ru_sample_rules() {
    let toml_str = load_text("rules/ru_sample.toml");
    let profile = LanguageRuleProfile::from_toml_str(&toml_str)
        .expect("Failed to parse ru_sample.toml");
    assert_eq!(profile.language, "ru");
    assert!(profile.word_boundary);

    assert!(profile.is_conditional_end("т.д."));
    assert!(profile.is_conditional_end("т. д."));
    assert!(profile.is_conditional_end("Т.Д."));
    assert!(profile.is_conditional_end("Т. Д."));
    assert!(profile.is_conditional_end("и др."));

    let ru_sent = "Книги, тетради и т. д.";
    assert!(profile.is_conditional_end_at(ru_sent, ru_sent.len() - 1));
    let ru_sent_nospace = "Книги, тетради и т.д.";
    assert!(profile.is_conditional_end_at(ru_sent_nospace, ru_sent_nospace.len() - 1));
}

#[test]
fn test_de_sample_rules() {
    let toml_str = load_text("rules/de_sample.toml");
    let profile = LanguageRuleProfile::from_toml_str(&toml_str)
        .expect("Failed to parse de_sample.toml");
    assert_eq!(profile.language, "de");
    assert!(profile.word_boundary);
    assert_eq!(profile.typography.quote_style.as_deref(), Some("german"));

    // Abbreviations
    assert!(profile.is_nonbreaking_prefix("Herr Dr.", 7, 5));
    assert!(profile.is_nonbreaking_prefix("wie z. B.", 8, 4));
    assert!(profile.is_conditional_end("usw."));
    assert!(profile.is_conditional_end("bzw."));

    // Conjunctions
    assert!(profile.matches_clause_conjunction(" aber es geht"));
    assert!(profile.matches_clause_conjunction(" obwohl das schwer ist"));
    assert!(!profile.matches_clause_conjunction(" abend ist schön"));

    // Discourse
    assert!(profile.is_discourse_chunk_starter("Er ist dort."));
    assert!(profile.is_discourse_chunk_starter("Deshalb machen wir das."));
}

#[test]
fn test_es_sample_rules() {
    let toml_str = load_text("rules/es_sample.toml");
    let profile = LanguageRuleProfile::from_toml_str(&toml_str)
        .expect("Failed to parse es_sample.toml");
    assert_eq!(profile.language, "es");
    assert!(profile.word_boundary);
    assert!(profile.typography.spanish_inverted);

    // Nonbreaking prefixes
    assert!(profile.is_nonbreaking_prefix("El Dr.", 5, 3));
    assert!(profile.is_nonbreaking_prefix("La Sra.", 6, 3));
    assert!(profile.is_conditional_end("pág."));

    // Conjunctions & discourse
    assert!(profile.matches_clause_conjunction(" pero no sabemos"));
    assert!(profile.matches_clause_conjunction(" sin embargo"));
    assert!(profile.is_discourse_chunk_starter("Él llegó a tiempo."));
    assert!(profile.is_discourse_chunk_starter("Además es importante."));

}

#[test]
fn test_fr_sample_rules() {
    let toml_str = load_text("rules/fr_sample.toml");
    let profile = LanguageRuleProfile::from_toml_str(&toml_str)
        .expect("Failed to parse fr_sample.toml");
    assert_eq!(profile.language, "fr");
    assert!(profile.word_boundary);
    assert!(profile.typography.french_spacing);
    assert_eq!(profile.typography.quote_style.as_deref(), Some("guillemets"));

    assert!(profile.is_nonbreaking_prefix("Bonjour M.", 9, 8));
    assert!(profile.is_nonbreaking_prefix("comme p. ex.", 11, 6));
    assert!(profile.matches_clause_conjunction(" mais oui"));
    assert!(profile.matches_clause_conjunction(" parce que c'est vrai"));
    assert!(profile.is_discourse_chunk_starter("Il fait beau."));
}

#[test]
fn test_zh_sample_rules() {
    let toml_str = load_text("rules/zh_sample.toml");
    let profile = LanguageRuleProfile::from_toml_str(&toml_str)
        .expect("Failed to parse zh_sample.toml");
    assert_eq!(profile.language, "zh");
    assert!(!profile.word_boundary);
    assert!(profile.typography.pangu_spacing);
    assert_eq!(profile.typography.quote_style.as_deref(), Some("curly"));
    assert_eq!(profile.syntax.clustering_gap_threshold, 3);
    assert_eq!(profile.syntax.head_direction, "head_initial");

    assert!(profile.matches_clause_conjunction(" 但是我们知道"));
    assert!(profile.matches_clause_conjunction(" 尽管如此"));
    assert!(profile.is_discourse_chunk_starter("他表示完全赞同。"));
    assert!(profile.is_discourse_chunk_starter("这是一项重大决定。"));
}
