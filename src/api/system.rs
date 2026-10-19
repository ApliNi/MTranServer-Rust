use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct VersionResponse {
    pub version: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
}

pub async fn get_version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn get_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}
