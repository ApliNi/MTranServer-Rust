//! Library crate root: re-exports the binary's modules so integration tests
//! and external callers can exercise the engine directly.

pub mod api;
pub mod config;
pub mod engine;
pub mod logger;
pub mod middleware;
pub mod models;
pub mod service;
pub mod ui;
pub mod utils;
pub mod version;

// Convenient top-level re-exports for testing and direct embedding
pub use config::ParseFormats;
pub use engine::{EngineConfig, EngineError, Model, SentencePiece, Shortlist, TranslationEngine};
pub use service::{
    translate, translate_batch, ServiceError, TranslatedText, TranslationOutput,
};
