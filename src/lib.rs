use anyhow::{Context, Result};
use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, PluginStartupPolicy, capability, plugin_server_info,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_BASE_URL: &str = "http://localhost:8000/v1";
const PLUGIN_ID: &str = "openai-endpoint";

fn upstream_base_url() -> String {
    std::env::var("MESH_LLM_PLUGIN_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

/// Reads the bearer key from a file path passed as `--api-key-file <path>`.
///
/// A file, not an env var or a bare CLI value, because mesh-llm's plugin
/// config has no generic env-passthrough channel to the child process (only
/// `command`/`args` reach it), and a bare arg would show up in `ps`/
/// `/proc/<pid>/cmdline`. Absent entirely, the plugin runs unauthenticated,
/// matching its original behavior against an open upstream.
fn api_key_from_args() -> Result<Option<String>> {
    let args: Vec<String> = std::env::args().collect();
    let Some(index) = args.iter().position(|arg| arg == "--api-key-file") else {
        return Ok(None);
    };
    let path = args
        .get(index + 1)
        .context("--api-key-file requires a path argument")?;
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("reading api key file at {path}"))?;
    let key = contents.trim().to_string();
    if key.is_empty() {
        anyhow::bail!("api key file at {path} is empty");
    }
    Ok(Some(key))
}

struct ProxyState {
    client: reqwest::Client,
    upstream_base_url: String,
    api_key: Option<String>,
}

/// Hop-by-hop / credential headers that must never ride through unchanged:
/// `host` is rebuilt from `upstream_base_url`, `authorization` from any
/// original caller is discarded so only this proxy's own key ever reaches
/// the upstream, and body-framing headers are recomputed by reqwest/axum
/// for the (possibly re-encoded) forwarded body.
fn is_hop_by_hop_request_header(name: &str) -> bool {
    matches!(
        name,
        "host" | "authorization" | "content-length" | "transfer-encoding" | "connection"
    )
}

fn is_hop_by_hop_response_header(name: &str) -> bool {
    matches!(name, "content-length" | "transfer-encoding" | "connection")
}

/// Path prefix this proxy always advertises as its own base (see
/// `spawn_auth_proxy`'s `format!("http://{addr}{ADVERTISED_PATH_PREFIX}")`).
/// Callers build requests by appending endpoint suffixes (`/models`,
/// `/chat/completions`) to whatever base_url we registered, so every incoming
/// request path starts with this prefix — it must be stripped before
/// re-appending the suffix to the real `upstream_base_url` (which carries
/// its own, possibly different, base path), or the two prefixes concatenate
/// into a broken URL like `.../v1/v1/models`.
const ADVERTISED_PATH_PREFIX: &str = "/v1";

async fn proxy_handler(State(state): State<Arc<ProxyState>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");
    let suffix = path_and_query
        .strip_prefix(ADVERTISED_PATH_PREFIX)
        .unwrap_or(path_and_query);
    let upstream_url = format!(
        "{}{}",
        state.upstream_base_url.trim_end_matches('/'),
        suffix
    );

    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("reading request body: {error}"),
            )
                .into_response();
        }
    };

    let mut upstream_request = state.client.request(parts.method, &upstream_url);
    for (name, value) in parts.headers.iter() {
        if is_hop_by_hop_request_header(name.as_str()) {
            continue;
        }
        upstream_request = upstream_request.header(name, value);
    }
    if let Some(key) = &state.api_key {
        upstream_request = upstream_request.bearer_auth(key);
    }

    let upstream_response = match upstream_request.body(body_bytes).send().await {
        Ok(response) => response,
        Err(error) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {error}"),
            )
                .into_response();
        }
    };

    let status = upstream_response.status();
    let mut response_headers = HeaderMap::new();
    for (name, value) in upstream_response.headers().iter() {
        if is_hop_by_hop_response_header(name.as_str()) {
            continue;
        }
        response_headers.insert(name.clone(), value.clone());
    }

    let mut response = Response::new(Body::from_stream(upstream_response.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    response
}

/// Binds a loopback-only reverse proxy that injects `Authorization: Bearer
/// <key>` (when configured) into every request before forwarding to the real
/// upstream. Returns the local address the plugin advertises to mesh-llm
/// instead of the real, possibly-credentialed upstream URL — so the bearer
/// key never appears in this process's manifest, in mesh gossip, or in any
/// other pool member's view of "where this model is served."
async fn spawn_auth_proxy(
    upstream_base_url: String,
    api_key: Option<String>,
) -> Result<SocketAddr> {
    let state = Arc::new(ProxyState {
        client: reqwest::Client::new(),
        upstream_base_url,
        api_key,
    });
    let app = Router::new().fallback(any(proxy_handler)).with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("binding local auth-proxy listener")?;
    let addr = listener.local_addr().context("reading local proxy addr")?;
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            eprintln!("openai-endpoint auth proxy stopped: {error:#}");
        }
    });
    Ok(addr)
}

fn build_plugin(
    name: String,
    advertised_base_url: String,
    upstream_base_url: String,
) -> mesh_llm_plugin::SimplePlugin {
    mesh_llm_plugin::plugin! {
        metadata: PluginMetadata::new(
            name,
            VERSION,
            plugin_server_info(
                "mesh-openai-endpoint",
                VERSION,
                "OpenAI-Compatible Endpoint Plugin",
                "Routes inference to an external OpenAI-compatible server (vLLM, TGI, \
                 Ollama, LM Studio, etc.), optionally authenticating to it with a \
                 bearer API key injected by a local proxy.",
                Some(
                    "Set MESH_LLM_PLUGIN_URL to point at any server that speaks the \
                     OpenAI /v1/chat/completions API. Pass --api-key-file <path> to a \
                     file containing a bearer token to authenticate to it; the token \
                     is injected by a local loopback proxy and never appears in this \
                     plugin's advertised endpoint address.",
                ),
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("endpoint:inference"),
            capability("endpoint:inference/openai_compatible"),
        ],
        inference: [
            mesh_llm_plugin::inference::openai_http(PLUGIN_ID, advertised_base_url.clone())
                .managed_by_plugin(false),
        ],
        health: move |_context| {
            let upstream_base_url = upstream_base_url.clone();
            Box::pin(async move { Ok(format!("upstream={upstream_base_url}")) })
        },
    }
}

async fn run_plugin(name: String) -> Result<()> {
    let upstream_base_url = upstream_base_url();
    let api_key = api_key_from_args()?;
    let proxy_addr = spawn_auth_proxy(upstream_base_url.clone(), api_key).await?;
    let advertised_base_url = format!("http://{proxy_addr}{ADVERTISED_PATH_PREFIX}");
    PluginRuntime::run(build_plugin(name, advertised_base_url, upstream_base_url)).await
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
    use anyhow::bail;
    use mesh_llm_plugin::Plugin;
    use serde_json::{Value, json};
    use std::time::Duration;

    fn loopback_manifest(advertised_base_url: &str) -> mesh_llm_plugin::proto::PluginManifest {
        let plugin = build_plugin(
            PLUGIN_ID.to_string(),
            advertised_base_url.to_string(),
            "http://real-upstream.example:1234/v1".to_string(),
        );
        plugin.manifest().expect("manifest")
    }

    #[test]
    fn manifest_declares_external_openai_endpoint() {
        let manifest = loopback_manifest("http://127.0.0.1:59123/v1");

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

    /// Permanent regression test for the property this fork exists to buy:
    /// whatever the real (possibly credentialed) upstream URL is, the
    /// manifest this plugin advertises to the mesh pool must always be a
    /// loopback address — never the real upstream, and never anything that
    /// could leak the API key's host/scheme to other pool members.
    #[test]
    fn advertised_endpoint_is_always_loopback_never_real_upstream() {
        for real_upstream in [
            "https://100.114.85.122:1234/v1",
            "https://lmstudio.tail637714.ts.net/v1",
            "http://localhost:8000/v1",
        ] {
            let plugin = build_plugin(
                PLUGIN_ID.to_string(),
                "http://127.0.0.1:59123/v1".to_string(),
                real_upstream.to_string(),
            );
            let manifest = plugin.manifest().expect("manifest");
            let address = manifest.endpoints[0]
                .address
                .as_deref()
                .expect("endpoint address");
            assert!(
                address.starts_with("http://127.0.0.1:"),
                "advertised endpoint {address} must be loopback, not derived from \
                 real upstream {real_upstream}"
            );
            assert_ne!(address, real_upstream);
        }
    }

    #[tokio::test]
    async fn proxy_injects_bearer_key_and_strips_caller_authorization() -> Result<()> {
        let upstream = axum_test_upstream().await?;
        // Real servers (e.g. LM Studio) are commonly configured with a base
        // URL that itself ends in "/v1" — exactly like this plugin's own
        // advertised base. Naively concatenating the two produces
        // "/v1/v1/models". Use a real "/v1" upstream base and an EXACT route
        // (no fallback) so a reintroduced double-prefix bug 404s instead of
        // silently passing.
        let proxy_addr = spawn_auth_proxy(
            format!("http://{}/v1", upstream.addr),
            Some("secret-lmstudio-key".to_string()),
        )
        .await?;

        let client = reqwest::Client::new();
        let response = client
            .get(format!("http://{proxy_addr}/v1/models"))
            .header("authorization", "Bearer caller-supplied-should-be-dropped")
            .send()
            .await?;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "must reach the upstream's exact /v1/models route, not /v1/v1/models"
        );
        let seen_auth = response.text().await?;
        assert_eq!(seen_auth, "Bearer secret-lmstudio-key");
        Ok(())
    }

    #[tokio::test]
    async fn proxy_returns_bad_gateway_when_upstream_unreachable() -> Result<()> {
        // Port 1 is reserved/unroutable, so this fails immediately without a
        // real dependency on "nothing listens there" being stable elsewhere.
        let proxy_addr = spawn_auth_proxy("http://127.0.0.1:1".to_string(), None).await?;
        let client = reqwest::Client::new();
        let response = client
            .get(format!("http://{proxy_addr}/v1/models"))
            .send()
            .await?;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        Ok(())
    }

    struct TestUpstream {
        addr: SocketAddr,
    }

    /// Minimal upstream double that echoes back the `Authorization` header
    /// it received, so tests can assert on exactly what the proxy sent.
    /// Routes ONLY `/v1/models` — deliberately no wildcard fallback, so a
    /// wrong forwarded path (e.g. a reintroduced "/v1/v1/models" double
    /// prefix) 404s instead of silently matching anyway.
    async fn axum_test_upstream() -> Result<TestUpstream> {
        async fn echo_auth(headers: HeaderMap) -> String {
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string()
        }
        let app = Router::new().route("/v1/models", axum::routing::get(echo_auth));
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(TestUpstream { addr })
    }

    #[tokio::test]
    async fn e2e_llama_server_answers_openai_requests() -> Result<()> {
        if std::env::var_os("OPENAI_ENDPOINT_E2E").is_none() {
            return Ok(());
        }

        let upstream_base_url =
            std::env::var("MESH_LLM_PLUGIN_URL").unwrap_or_else(|_| upstream_base_url());
        let proxy_addr = spawn_auth_proxy(upstream_base_url.clone(), api_key_from_args()?).await?;
        let base_url = format!("http://{proxy_addr}/v1");

        let manifest = loopback_manifest(&base_url);
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
