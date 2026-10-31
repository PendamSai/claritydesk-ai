"""Server-owned model configuration for local and hosted deployments."""
import os
from urllib.parse import urlsplit


def public_deployment():
    return os.getenv("PUBLIC_DEPLOYMENT", "false").lower() in ("1", "true", "yes")


def ollama_settings():
    endpoint = os.getenv("OLLAMA_BASE_URL", "").strip()
    if not endpoint:
        return {}
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("OLLAMA_BASE_URL must be a valid HTTP or HTTPS server endpoint.")
    return {"base_url": endpoint}
