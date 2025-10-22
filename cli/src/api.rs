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

    pub async fn health_check(&self) -> Result<()> {
        let url = self.base_url.join("health").context("failed to build health URL")?;
        let response = self.http.get(url).send().await.context("worker health request failed")?;
        if !response.status().is_success() {
            anyhow::bail!("worker responded with status {}", response.status());
        }
        Ok(())
    }

    pub fn base_url(&self) -> &Url {
        &self.base_url
    }
}
