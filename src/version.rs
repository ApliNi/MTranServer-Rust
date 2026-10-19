//! Web UI asset version, defined once here.
//!
//! The HTML `<base href="/${UI_VERSION}/">` and the versioned asset route
//! prefix both derive from it. Bump it whenever any embedded UI asset
//! changes to invalidate clients' one-year asset caches.

pub const UI_VERSION: &str = "v0.11";
