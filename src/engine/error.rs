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

    /// I/O error during file read or access.
    Io(String),

    /// Worker or loader thread panicked during execution.
    WorkerPanicked(String),

    /// Other engine failure.
    Other(String),
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
