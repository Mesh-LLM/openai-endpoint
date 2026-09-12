//! Exercise the executable's control connection, not just its declared URL.
#![cfg(unix)]

use anyhow::{Context, Result};
use mesh_llm_plugin::{LocalStream, proto, read_envelope, write_envelope};
use std::time::Duration;
use tokio::{net::UnixListener, process::Command, time::timeout};

#[tokio::test]
async fn executable_initializes_with_mesh_076_protocol() -> Result<()> {
    // Keep the socket path short enough for macOS, regardless of checkout path.
    let directory = tempfile::Builder::new().prefix("ep-").tempdir_in("/tmp")?;
    let socket = directory.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let mut child = Command::new(env!("CARGO_BIN_EXE_openai-endpoint"))
        .env("MESH_LLM_PLUGIN_ENDPOINT", &socket)
        .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
        .env("MESH_LLM_PLUGIN_URL", "http://127.0.0.1:8000/v1")
        .kill_on_drop(true)
        .spawn()?;

    let outcome = timeout(Duration::from_secs(10), async {
        let (socket, _) = listener.accept().await?;
        let mut stream = LocalStream::Unix(socket);
        write_envelope(
            &mut stream,
            &proto::Envelope {
                // Pin the consumer contract independently of the SDK constant.
                protocol_version: 3,
                plugin_id: "openai-endpoint".into(),
                request_id: 1,
                payload: Some(proto::envelope::Payload::InitializeRequest(
                    proto::InitializeRequest {
                        host_protocol_version: 3,
                        host_version: "0.76.0".into(),
                        host_info_json: "{}".into(),
                        mesh_visibility: proto::MeshVisibility::Private as i32,
                    },
                )),
            },
        )
        .await?;
        let response = read_envelope(&mut stream).await?;
        anyhow::ensure!(response.protocol_version == 3, "wrong envelope protocol");
        anyhow::ensure!(response.request_id == 1, "wrong response request ID");
        let Some(proto::envelope::Payload::InitializeResponse(init)) = response.payload else {
            anyhow::bail!("expected successful initialize response");
        };
        anyhow::ensure!(
            init.plugin_protocol_version == 3,
            "incompatible plugin protocol"
        );
        anyhow::ensure!(
            init.plugin_id == "openai-endpoint",
            "changed install identity"
        );
        let manifest = init.manifest.context("missing manifest")?;
        anyhow::ensure!(
            manifest.endpoints.len() == 1,
            "expected one external endpoint"
        );
        let endpoint = &manifest.endpoints[0];
        anyhow::ensure!(
            endpoint.address.as_deref() == Some("http://127.0.0.1:8000/v1"),
            "configured endpoint URL was not propagated"
        );
        anyhow::ensure!(!endpoint.managed_by_plugin, "must not manage the upstream");
        Ok::<_, anyhow::Error>(())
    })
    .await;

    // Reap our own child even when the protocol assertion fails.
    child.kill().await?;
    child.wait().await?;
    outcome.context("plugin handshake timed out")?
}
