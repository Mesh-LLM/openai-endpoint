# openai-endpoint

`openai-endpoint` is an external mesh-llm plugin for routing inference to an
already-running OpenAI-compatible server such as vLLM, TGI, Ollama, or Lemonade
Server.

## Install

```bash
mesh-llm plugins install openai-endpoint
```

You can also install directly from GitHub:

```bash
mesh-llm plugins install Mesh-LLM/openai-endpoint
```

## Configure

Point the plugin at one external server:

```toml
[[plugin]]
name = "openai-endpoint"
url = "http://localhost:8000/v1"
```

If you are running the binary yourself instead of installing it through
`mesh-llm plugins install`, provide the command explicitly:

```toml
[[plugin]]
name = "openai-endpoint"
command = "openai-endpoint"
url = "http://localhost:8000/v1"
```

mesh-llm passes `url` to the plugin as `MESH_LLM_PLUGIN_URL`. If neither config
nor environment is set, it defaults to `http://localhost:8000/v1`.

### Multiple endpoints

The same plugin instance can advertise one or more OpenAI-compatible servers.
The existing single-URL form remains supported. For multiple servers, use
`urls`:

```toml
[[plugin]]
name = "openai-endpoint"
urls = [
  "http://localhost:8000/v1",
  "http://gpu-box:8000/v1",
]
```

If `urls` is set, the plugin receives the values through
`MESH_LLM_PLUGIN_URLS`. A single endpoint keeps the endpoint ID
`openai-endpoint`; multiple endpoints are assigned `openai-endpoint-1`,
`openai-endpoint-2`, and so on. Newline-separated URLs are also accepted when
setting the environment variable directly. Each endpoint is health-checked
independently, and the mesh routes a model to the healthy endpoint that
advertises it.

### Authentication suggestions

Authentication is not currently applied by this plugin. The plugin advertises
the endpoint address, while the mesh host performs the model health probe and
inference request, so adding a token only inside this process would not secure
those host-side requests.

Recommended options, in order of practicality:

1. Add host-managed per-endpoint credentials. Extend the endpoint configuration
   with an auth reference such as `bearer_env = "GPU_BOX_API_KEY"`, keep the
   secret out of the manifest, and have the host attach the header to both
   `/v1/models` health probes and inference requests. This supports different
   credentials per endpoint and secret rotation without exposing tokens in
   URLs, logs, or endpoint metadata.
2. Use a local authenticated gateway or sidecar today. Point this plugin at
   Envoy, an OAuth2 proxy, or another local gateway that injects credentials,
   and let the gateway forward to the protected upstream. This requires no
   mesh protocol change and keeps credentials outside the plugin URL.
3. For OAuth2/OIDC, add a host-side token provider with caching, expiry-aware
   refresh, and separate scopes per endpoint. The same provider should be used
   by health checks and inference forwarding.

Avoid putting bearer tokens in query strings or URL user-info, since endpoint
addresses can appear in manifests, diagnostics, and logs.

## Build

```bash
cargo build
```

## Release Archives

GitHub releases package archives using the same contract as
`Mesh-LLM/blackboard`:

- `openai-endpoint-<target>.tar.gz` or `.zip`
- `openai-endpoint-<version>-<target>.tar.gz` or `.zip`

Each archive contains:

- `openai-endpoint/openai-endpoint`
- `openai-endpoint/plugin.toml`
