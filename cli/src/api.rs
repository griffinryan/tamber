use crate::types::{GenerationArtifact, GenerationRequest, GenerationStatus};
use anyhow::{Context, Result};
use reqwest::Url;

const DEFAULT_BASE_URL: &str = "http://localhost:8000";

pub struct Client {
    http: reqwest::Client,
    base_url: Url,
}

impl Client {
    pub fn new(base_url: Option<&str>) -> Result<Self> {
        let url = base_url
            .map(Url::parse)
            .unwrap_or_else(|| Url::parse(DEFAULT_BASE_URL))
            .context("invalid worker base URL")?;
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self { http, base_url: url })
    }

    pub async fn health(&self) -> Result<serde_json::Value> {
        let url = self.base_url.join("health").context("failed to build health URL")?;
        let response = self.http.get(url).send().await.context("worker health request failed")?;
        if !response.status().is_success() {
            anyhow::bail!("worker responded with status {}", response.status());
        }
        response.json().await.context("failed to decode health response")
    }

    pub async fn submit_generation(&self, request: &GenerationRequest) -> Result<GenerationStatus> {
        let url = self.base_url.join("generate").context("failed to build generate URL")?;
        let response = self
            .http
            .post(url)
            .json(request)
            .send()
            .await
            .context("failed to submit generation request")?;
        if !response.status().is_success() {
            anyhow::bail!("generation submission failed with status {}", response.status());
        }
        response.json().await.context("failed to decode generation response")
    }

    pub async fn job_status(&self, job_id: &str) -> Result<GenerationStatus> {
        let url = self
            .base_url
            .join(&format!("status/{job_id}"))
            .context("failed to build status URL")?;
        let response = self.http.get(url).send().await.context("failed to query job status")?;
        if !response.status().is_success() {
            anyhow::bail!("status poll failed with status {}", response.status());
        }
        response.json().await.context("failed to decode status response")
    }

    pub async fn fetch_artifact(&self, job_id: &str) -> Result<GenerationArtifact> {
        let url = self
            .base_url
            .join(&format!("artifact/{job_id}"))
            .context("failed to build artifact URL")?;
        let response =
            self.http.get(url).send().await.context("failed to fetch artifact metadata")?;
        if !response.status().is_success() {
            anyhow::bail!("artifact fetch failed with status {}", response.status());
        }
        response.json().await.context("failed to decode artifact response")
    }

    pub fn base_url(&self) -> &Url {
        &self.base_url
    }
}

impl Clone for Client {
    fn clone(&self) -> Self {
        Self { http: self.http.clone(), base_url: self.base_url.clone() }
    }
}
