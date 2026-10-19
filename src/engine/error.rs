//! Engine error types for Marian/Bergamot NMT translation engine.

use std::fmt;

/// Strongly-typed domain error for the translation engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// Format error in binary model, shortlist, or vocabulary files
    /// (corrupted headers, unexpected EOF, magic mismatch, invalid version, etc.).
    Format(String),

    /// Dimension mismatch or invalid tensor dimensions.
    DimensionMismatch(String),

    /// Vocabulary file missing, corrupt, or missing expected pieces.
    VocabularyMissing(String),

    /// Model file not loaded, missing, or failed to initialize.
    ModelNotLoaded(String),

    /// I/O error during file read or access.
    Io(String),

    /// Worker or loader thread panicked during execution.
    WorkerPanicked(String),

    /// Other engine failure.
    Other(String),
}

impl EngineError {
    pub fn format(msg: impl Into<String>) -> Self {
        EngineError::Format(msg.into())
    }

    pub fn dimension_mismatch(msg: impl Into<String>) -> Self {
        EngineError::DimensionMismatch(msg.into())
    }

    pub fn vocabulary_missing(msg: impl Into<String>) -> Self {
        EngineError::VocabularyMissing(msg.into())
    }

    pub fn model_not_loaded(msg: impl Into<String>) -> Self {
        EngineError::ModelNotLoaded(msg.into())
    }

    pub fn io(msg: impl Into<String>) -> Self {
        EngineError::Io(msg.into())
    }

    pub fn worker_panicked(msg: impl Into<String>) -> Self {
        EngineError::WorkerPanicked(msg.into())
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Format(msg) => {
                if msg.starts_with("Format error") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "Format error: {msg}")
                }
            }
            EngineError::DimensionMismatch(msg) => {
                if msg.starts_with("Dimension mismatch") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "Dimension mismatch: {msg}")
                }
            }
            EngineError::VocabularyMissing(msg) => {
                if msg.starts_with("Vocabulary missing") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "Vocabulary missing: {msg}")
                }
            }
            EngineError::ModelNotLoaded(msg) => {
                if msg.starts_with("Model not loaded") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "Model not loaded: {msg}")
                }
            }
            EngineError::Io(msg) => {
                if msg.starts_with("I/O error") || msg.starts_with("Failed to read") {
                    write!(f, "{msg}")
                } else {
                    write!(f, "I/O error: {msg}")
                }
            }
            EngineError::WorkerPanicked(msg) => write!(f, "Worker panicked: {msg}"),
            EngineError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(err: std::io::Error) -> Self {
        EngineError::Io(err.to_string())
    }
}

impl From<String> for EngineError {
    fn from(s: String) -> Self {
        EngineError::Other(s)
    }
}

impl From<&str> for EngineError {
    fn from(s: &str) -> Self {
        EngineError::Other(s.to_string())
    }
}

impl From<EngineError> for String {
    fn from(err: EngineError) -> Self {
        err.to_string()
    }
}
