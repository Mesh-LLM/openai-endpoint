use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{
    DeclarativePluginBuilder, PluginMetadata, PluginRuntime, PluginStartupPolicy, capability,
    plugin_server_info,
};
use serde_json::Value;
use std::collections::HashSet;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";
const PLUGIN_ID: &str = "openai-endpoint";

#[derive(Clone, Debug, PartialEq, Eq)]
struct EndpointSpec {
    id: String,
    url: String,
}

fn configured_endpoint_specs() -> Result<Vec<EndpointSpec>> {
    let value = std::env::var("MESH_LLM_PLUGIN_URLS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var("MESH_LLM_PLUGIN_URL")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    endpoint_specs_from_value(&value)
}

/// Parse the value passed through the plugin URL environment variables.
///
/// A plain URL remains the single-endpoint format. For multiple endpoints,
/// use a JSON array of strings or objects with an optional stable `id`:
/// `["http://one/v1", {"id":"two", "url":"http://two/v1"}]`.
fn endpoint_specs_from_value(value: &str) -> Result<Vec<EndpointSpec>> {
    let value = value.trim();
    if value.is_empty() {
        bail!("at least one endpoint URL must be configured");
    }

    let urls = if value.starts_with('[') {
        let entries = serde_json::from_str::<Value>(value)
            .context("parse plugin JSON endpoint list")?
            .as_array()
            .cloned()
            .context("plugin JSON URL value must be an array")?;
        entries
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                let (id, url) = match entry {
                    Value::String(url) => (None, url),
                    Value::Object(object) => {
                        let id = object.get("id").and_then(Value::as_str).map(str::to_string);
                        let url = object
                            .get("url")
                            .and_then(Value::as_str)
                            .with_context(|| {
                                format!("endpoint {index} object must contain a string `url`")
                            })?
                            .trim()
                            .to_string();
                        (id, url)
                    }
                    _ => bail!("endpoint {index} must be a URL string or an object"),
                };
                Ok((id, url))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        // Newlines are accepted for convenient environment-variable based
        // configuration while keeping commas valid inside URLs.
        value
            .lines()
            .map(|url| (None, url.trim().to_string()))
            .filter(|(_, url)| !url.is_empty())
            .collect()
    };

    if urls.is_empty() {
        bail!("at least one endpoint URL must be configured");
    }

    let endpoint_count = urls.len();
    let specs = urls
        .into_iter()
        .enumerate()
        .map(|(index, (id, url))| {
            let url = url.trim().to_string();
            if url.is_empty() {
                bail!("endpoint {index} URL must not be empty");
            }
            let id = id.unwrap_or_else(|| {
                if endpoint_count == 1 {
                    PLUGIN_ID.to_string()
                } else {
                    format!("{PLUGIN_ID}-{}", index + 1)
                }
            });
            let id = id.trim().to_string();
            if id.is_empty() {
                bail!("endpoint {index} ID must not be empty");
            }
            Ok(EndpointSpec { id, url })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut ids = HashSet::new();
    for spec in &specs {
        if !ids.insert(&spec.id) {
            bail!("duplicate endpoint ID '{}'", spec.id);
        }
    }
    Ok(specs)
}

fn build_plugin(name: String) -> Result<mesh_llm_plugin::SimplePlugin> {
    let endpoints = configured_endpoint_specs()?;
    let endpoint_summary = endpoints
        .iter()
        .map(|endpoint| format!("{}={}", endpoint.id, endpoint.url))
        .collect::<Vec<_>>()
        .join(", ");

    let metadata = PluginMetadata::new(
        name,
        VERSION,
        plugin_server_info(
            "mesh-openai-endpoint",
            VERSION,
            "OpenAI-Compatible Endpoint Plugin",
            "Routes inference to external OpenAI-compatible servers (vLLM, TGI, Ollama, etc.).",
            Some(
                "Set MESH_LLM_PLUGIN_URL for one endpoint or MESH_LLM_PLUGIN_URLS \
                 for multiple endpoint URLs.",
            ),
        ),
    );
    let mut builder = DeclarativePluginBuilder::new(metadata)
        .startup_policy(PluginStartupPolicy::Any)
        .provide(capability("endpoint:inference"))
        .provide(capability("endpoint:inference/openai_compatible"));
    for endpoint in endpoints {
        builder = builder.inference_item(
            mesh_llm_plugin::inference::openai_http(endpoint.id, endpoint.url)
                .managed_by_plugin(false),
        );
    }
    builder = builder.customize(move |plugin| {
        plugin.with_health(move |_context| {
            let endpoint_summary = endpoint_summary.clone();
            Box::pin(async move { Ok(format!("endpoints={endpoint_summary}")) })
        })
    });
    Ok(builder.build())
}

async fn run_plugin(name: String) -> Result<()> {
    PluginRuntime::run(build_plugin(name)?).await
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
        let plugin = build_plugin(PLUGIN_ID.to_string()).expect("build plugin");
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

        let base_url =
            std::env::var("MESH_LLM_PLUGIN_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let plugin = build_plugin(PLUGIN_ID.to_string())?;
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

    #[test]
    fn endpoint_urls_keep_single_endpoint_id_compatible() {
        assert_eq!(
            endpoint_specs_from_value("http://localhost:8000/v1").unwrap(),
            vec![EndpointSpec {
                id: PLUGIN_ID.to_string(),
                url: "http://localhost:8000/v1".to_string(),
            }]
        );
    }

    #[test]
    fn endpoint_urls_accept_json_array_with_optional_ids() {
        let specs = endpoint_specs_from_value(
            r#"["http://one:8000/v1", {"id":"remote", "url":"https://two.example/v1"}]"#,
        )
        .unwrap();
        assert_eq!(specs[0].id, "openai-endpoint-1");
        assert_eq!(specs[0].url, "http://one:8000/v1");
        assert_eq!(specs[1].id, "remote");
        assert_eq!(specs[1].url, "https://two.example/v1");
    }

    #[test]
    fn endpoint_urls_accept_newline_separated_values() {
        let specs = endpoint_specs_from_value(" http://one/v1\n\nhttp://two/v1 ").unwrap();
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.url.as_str())
                .collect::<Vec<_>>(),
            vec!["http://one/v1", "http://two/v1"]
        );
    }

    #[test]
    fn endpoint_urls_reject_duplicate_ids() {
        let error = endpoint_specs_from_value(
            r#"[{"id":"same", "url":"http://one/v1"}, "http://two/v1", {"id":"same", "url":"http://three/v1"}]"#,
        )
        .expect_err("duplicate IDs should be rejected");
        assert!(error.to_string().contains("duplicate endpoint ID 'same'"));
    }
}
