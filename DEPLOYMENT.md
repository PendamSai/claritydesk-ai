# Public website deployment

Status: application container and hosted-runtime configuration prepared. No hosting resources or public URL have been created. This is a single-server pilot, not an authenticated multi-user production service.

## Container hosting

Build with `docker build -t ai-assistant .`. The image runs Streamlit as a non-root user on port 8501. Its health endpoint is `/_stcore/health`. Host it behind the hosting provider's HTTPS ingress; keep the model service private. Do not expose Ollama port 11434 publicly.

Set server environment variables:

- `PUBLIC_DEPLOYMENT=true`: disables visitor model-name changes, limits speech model selection to tiny, and changes privacy text to disclose server processing.
- `OLLAMA_BASE_URL`: private HTTP(S) model endpoint. A host:port value is accepted and gets an HTTP prefix. All document embeddings, chat, meeting notes, research and standalone verification use this endpoint. The endpoint must have the configured models installed.
- `CHAT_MODEL=llama3.2`
- `EMBEDDING_MODEL=nomic-embed-text`

The hosting provider must supply enough RAM and CPU for the web process, speech transcription and model inference. Model inference on a CPU can still take minutes. More simultaneous visitors require capacity controls and appropriate infrastructure. Model files download on first use; a persistent speech cache avoids repeat downloads across restarts.

## Self-hosted pilot

On a machine with Docker Compose, run `docker compose up --build -d`. This starts a private Ollama service, downloads its two models, then starts the app. Only Streamlit is bound to the host, at `127.0.0.1:8501`. Add your hosting platform's HTTPS ingress or a reverse proxy before public access. The Compose file does not open a public internet port, configure a domain, or provision a server. Persistent named volumes hold model files, not uploaded user documents.

## Visitor data and public launch limits

Document indexes, transcripts, reports, and chats remain in Streamlit per-session state. They are not account-owned durable history, and reconnects/restarts can lose data. Speech model resources are shared; user transcripts are not put into a shared application cache. Hosted uploads leave the visitor's device and are processed by your server; web-enabled research also sends objectives and generated queries to external search providers. Do not advertise on-device processing in a hosted deployment.

The Docker build context allowlist excludes `.env`, local recordings, user documents, virtual environments and tests. Uploads are capped at 100 MB per file. Before an unrestricted public launch, add suitable account access, retention/deletion rules, aggregate storage limits, per-user quotas and inference concurrency controls. The existing session caches and research-step limit are not abuse protection or billing caps.

## Publishing

Connect a hosting account that supports long-running Python/Docker services, choose its compute budget, provision a private model endpoint, then deploy the container. Render and Railway integrations were found but are not connected in this session. No paid resources have been created. Docker is not installed in the current workspace, so the container build has not been executed here.
