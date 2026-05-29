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

Point the plugin at the external server:

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
