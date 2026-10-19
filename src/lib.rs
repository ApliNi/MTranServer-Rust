//! Library crate root: re-exports the binary's modules so integration tests
//! and benchmarks can exercise the engine directly.

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
