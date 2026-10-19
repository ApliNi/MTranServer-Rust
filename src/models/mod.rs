//! Model records management and download.
//!
//! Mirrors `src/models/records.ts` in the original project. Downloads model
//! metadata from Mozilla's Firefox Translations service and fetches the actual
//! model files (.zst compressed) into the model directory.

pub mod download;
pub mod records;
pub mod updater;

pub use records::{
    download_model, get_language_pairs, get_model_files, get_supported_languages,
    has_language_pair, init_records, is_supported_language,
};
