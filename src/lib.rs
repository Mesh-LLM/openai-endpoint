use anyhow::Result;
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, PluginStartupPolicy, capability, plugin_server_info,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";
const PLUGIN_ID: &str = "openai-endpoint";

fn base_url() -> String {
    std::env::var("MESH_LLM_PLUGIN_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

fn build_plugin(name: String) -> mesh_llm_plugin::SimplePlugin {
    let base_url = base_url();
    let health_url = base_url.clone();

    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            VERSION,
            plugin_server_info(
                "mesh-openai-endpoint",
                VERSION,
                "OpenAI-Compatible Endpoint Plugin",
                "Routes inference to an external OpenAI-compatible server (vLLM, TGI, Ollama, etc.).",
                Some(
                    "Set MESH_LLM_PLUGIN_URL to point at any server \
                     that speaks the OpenAI /v1/chat/completions API.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
        ],
        inference: [
            mesh_llm_plugin::inference::openai_http(PLUGIN_ID, base_url.clone())
                .managed_by_plugin(false),
        ],
        health: move |_context| {
            let health_url = health_url.clone();
            Box::pin(async move { Ok(format!("base_url={health_url}")) })
        },
    }
}

async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_plugin(name)).await
}

pub fn run_main() -> i32 {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    let runtime = builder.build().expect("build tokio runtime");
    runtime.block_on(async move {
        match run_plugin(PLUGIN_ID.to_string()).await {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("{err:#}");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, bail};
    use mesh_llm_plugin::Plugin;
    use serde_json::{Value, json};
    use std::time::Duration;

    #[test]
    fn manifest_declares_external_openai_endpoint() {
        let plugin = build_plugin(PLUGIN_ID.to_string());
        let manifest = plugin.manifest().expect("manifest");

        assert!(
            manifest
                .capabilities
                .iter()
                .any(|capability| capability == "endpoint:inference/openai_compatible")
        );
        assert_eq!(manifest.endpoints.len(), 1);
        assert_eq!(manifest.endpoints[0].endpoint_id, PLUGIN_ID);
        assert!(!manifest.endpoints[0].managed_by_plugin);
    }

    #[tokio::test]
    async fn e2e_llama_server_answers_openai_requests() -> Result<()> {
        if std::env::var_os("OPENAI_ENDPOINT_E2E").is_none() {
            return Ok(());
        }

        let base_url = std::env::var("MESH_LLM_PLUGIN_URL").unwrap_or_else(|_| base_url());
        let plugin = build_plugin(PLUGIN_ID.to_string());
        let manifest = plugin.manifest().context("plugin manifest")?;
        let endpoint = manifest
            .endpoints
            .iter()
            .find(|endpoint| endpoint.endpoint_id == PLUGIN_ID)
            .context("openai endpoint manifest entry")?;

        assert_eq!(endpoint.address.as_deref(), Some(base_url.as_str()));
        assert!(!endpoint.managed_by_plugin);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        let model = wait_for_model(&client, &base_url).await?;
        let body = client
            .post(endpoint_url(&base_url, "chat/completions"))
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": "Reply with one short word."}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": false
            }))
            .send()
            .await
            .context("send chat completion")?;
        let status = body.status();
        let value = body.json::<Value>().await.context("parse chat response")?;
        if !status.is_success() {
            bail!("chat completion failed with {status}: {value}");
        }
        let content = value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if content.is_empty() {
            bail!("chat completion returned empty content: {value}");
        }
        Ok(())
    }

    async fn wait_for_model(client: &reqwest::Client, base_url: &str) -> Result<String> {
        let mut last_error = None;
        for _ in 0..60 {
            match fetch_first_model(client, base_url).await {
                Ok(model) => return Ok(model),
                Err(error) => last_error = Some(error),
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("llama-server did not become ready")))
    }

    async fn fetch_first_model(client: &reqwest::Client, base_url: &str) -> Result<String> {
        let response = client
            .get(endpoint_url(base_url, "models"))
            .send()
            .await
            .context("fetch models")?;
        let status = response.status();
        let value = response.json::<Value>().await.context("parse models")?;
        if !status.is_success() {
            bail!("models request failed with {status}: {value}");
        }
        value
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find_map(|entry| entry.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .context("no model ids in /v1/models response")
    }

    fn endpoint_url(base_url: &str, tail: &str) -> String {
        format!("{}/{}", base_url.trim_end_matches('/'), tail)
    }
}
