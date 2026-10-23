use std::path::PathBuf;
use std::process;

use clap::Parser;
use mtranserver_test::regression::{
    default_corpus_path, init_test_environment, update_cases_file,
};

/// Re-translates regression test cases and updates `test/cases/regression.json` in place.
#[derive(Parser, Debug)]
#[command(
    name = "regression",
    about = "Re-translates test cases and updates the JSON file; inspect diffs with `git diff`"
)]
struct Args {
    /// Custom path to regression cases JSON file (defaults to test/cases/regression.json)
    #[arg(short, long)]
    corpus: Option<PathBuf>,

    /// Run with mock translation engine for fast testing without local Marian models
    #[arg(long)]
    mock: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let corpus_path = args.corpus.unwrap_or_else(default_corpus_path);

    println!("============================================================");
    println!("     MTranServer Translation Quality Regression Runner     ");
    println!("============================================================");
    println!("Corpus File: {:?}", corpus_path);
    println!("Mock Mode:   {}", args.mock);
    println!("------------------------------------------------------------");

    if !args.mock {
        println!("Initializing translation models and runtime...");
        if let Err(e) = init_test_environment().await {
            eprintln!("Failed to initialize test environment: {}", e);
            eprintln!(
                "Note: Ensure local Marian models exist in data/models/, or use --mock for fast CI tests."
            );
            process::exit(1);
        }
    }

    println!("Translating and updating {:?}...", corpus_path);
    match update_cases_file(&corpus_path, args.mock).await {
        Ok(count) => {
            println!(
                "\n[OK] Successfully updated {} test case(s) in {:?}",
                count, corpus_path
            );
            println!(
                "Inspect translation quality differences with:\n  git diff {:?}",
                corpus_path
            );
        }
        Err(e) => {
            eprintln!("Error updating regression cases: {}", e);
            process::exit(1);
        }
    }
}
