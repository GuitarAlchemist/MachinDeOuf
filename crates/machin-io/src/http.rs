//! HTTP client for fetching data from REST APIs.

use crate::error::IoError;
use crate::protocol::DataBatch;

/// Fetch JSON data from a URL and parse into DataBatch.
pub async fn fetch_json(url: &str) -> Result<DataBatch, IoError> {
    let response = reqwest::get(url).await?;
    let text = response.text().await?;
    crate::json_io::read_json_string(&text)
}

/// Fetch CSV data from a URL.
pub async fn fetch_csv(url: &str, has_header: bool) -> Result<DataBatch, IoError> {
    let response = reqwest::get(url).await?;
    let text = response.text().await?;
    crate::csv_io::read_csv_string(&text, has_header)
}

/// POST JSON data to a URL and get a response batch.
pub async fn post_json(url: &str, batch: &DataBatch) -> Result<DataBatch, IoError> {
    let json_str = crate::json_io::batch_to_json(batch)?;
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(json_str)
        .send()
        .await?;

    let text = response.text().await?;
    crate::json_io::read_json_string(&text)
}

/// Fetch raw bytes from a URL.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, IoError> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    Ok(bytes.to_vec())
}

/// Fetch NDJSON (newline-delimited JSON) from a URL.
pub async fn fetch_ndjson(url: &str) -> Result<DataBatch, IoError> {
    let response = reqwest::get(url).await?;
    let text = response.text().await?;
    crate::json_io::read_ndjson_string(&text)
}

/// HTTP data source configuration.
pub struct HttpSource {
    pub url: String,
    pub method: HttpMethod,
    pub headers: Vec<(String, String)>,
    pub poll_interval_secs: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum HttpMethod {
    Get,
    Post { body: String },
}

impl HttpSource {
    pub fn get(url: &str) -> Self {
        Self {
            url: url.to_string(),
            method: HttpMethod::Get,
            headers: Vec::new(),
            poll_interval_secs: None,
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.push((key.to_string(), value.to_string()));
        self
    }

    /// Poll the URL at regular intervals and send data through a channel.
    pub async fn poll(
        self,
        interval_secs: u64,
    ) -> Result<tokio::sync::mpsc::Receiver<DataBatch>, IoError> {
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let client = reqwest::Client::new();

        tokio::spawn(async move {
            loop {
                let mut request = match &self.method {
                    HttpMethod::Get => client.get(&self.url),
                    HttpMethod::Post { body } => client
                        .post(&self.url)
                        .header("Content-Type", "application/json")
                        .body(body.clone()),
                };

                for (key, value) in &self.headers {
                    request = request.header(key, value);
                }

                if let Ok(response) = request.send().await {
                    if let Ok(text) = response.text().await {
                        if let Ok(batch) = crate::json_io::read_json_string(&text) {
                            if tx.send(batch).await.is_err() {
                                break;
                            }
                        }
                    }
                }

                tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)).await;
            }
        });

        Ok(rx)
    }
}
