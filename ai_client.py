"""Small Gemini REST adapter; local Ollama stays available without cloud credentials."""
import json
import os
import re
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from runtime_config import ollama_settings


class AIServiceError(ValueError):
    pass


def ai_provider():
    provider = os.getenv("AI_PROVIDER", "ollama").strip().lower()
    if provider not in ("ollama", "gemini"):
        raise AIServiceError("AI_PROVIDER must be ollama or gemini.")
    return provider


def default_chat_model():
    if ai_provider() == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return os.getenv("CHAT_MODEL", "llama3.2")


_gate = threading.BoundedSemaphore(2)
_usage_lock = threading.Lock()
_usage_day = ""
_usage_count = 0


def _reserve_request():
    global _usage_day, _usage_count
    day = datetime.now(timezone.utc).date().isoformat()
    try:
        cap = max(1, min(1000, int(os.getenv("GEMINI_DAILY_REQUEST_LIMIT", "100"))))
    except ValueError:
        raise AIServiceError("GEMINI_DAILY_REQUEST_LIMIT must be an integer.") from None
    with _usage_lock:
        if day != _usage_day:
            _usage_day, _usage_count = day, 0
        if _usage_count >= cap:
            raise AIServiceError("This app has reached its daily AI request allowance. Please try another day.")
        _usage_count += 1


class GeminiChat:
    def __init__(self, model, num_predict=512, temperature=0, format=None, **ignored):
        self.key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.key:
            raise AIServiceError("Gemini is not configured. The app owner must add GEMINI_API_KEY in Streamlit Secrets.")
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
            raise AIServiceError("Invalid Gemini model name.")
        self.model = model
        self.generation = {"temperature": temperature, "maxOutputTokens": min(int(num_predict), 1500)}
        if format == "json":
            self.generation["responseMimeType"] = "application/json"

    def _request(self, prompt, streaming):
        if not isinstance(prompt, str) or len(prompt) > 60000:
            raise AIServiceError("This request is too long. Use fewer source passages or a shorter transcript.")
        _reserve_request()
        method = "streamGenerateContent?alt=sse" if streaming else "generateContent"
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:{method}"
        body = {"contents": [{"role":"user", "parts":[{"text":prompt}]}], "generationConfig":self.generation}
        return Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type":"application/json", "x-goog-api-key":self.key}, method="POST")

    @staticmethod
    def _text(payload):
        if payload.get("error"):
            raise AIServiceError("The AI service could not complete this request. No automatic retry was made.")
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        return "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []) if not p.get("thought"))

    def _run(self, prompt, streaming):
        if not _gate.acquire(blocking=False):
            raise AIServiceError("The free app is busy. Please try again shortly.")
        try:
            request = self._request(prompt, streaming)
            with urlopen(request, timeout=90) as response:
                if not streaming:
                    text = self._text(json.loads(response.read()))
                    if not text.strip():
                        raise AIServiceError("The AI service returned no answer or blocked this request. Try rephrasing it.")
                    yield SimpleNamespace(content=text)
                    return
                data_lines, received = [], False
                for raw in response:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        payload = "\n".join(data_lines)
                        data_lines = []
                        if payload == "[DONE]":
                            continue
                        text = self._text(json.loads(payload))
                        if text:
                            received = True
                            yield SimpleNamespace(content=text)
                if data_lines:
                    text = self._text(json.loads("\n".join(data_lines)))
                    if text:
                        received = True
                        yield SimpleNamespace(content=text)
                if not received:
                    raise AIServiceError("The AI service returned no answer or blocked this request. Try rephrasing it.")
        except HTTPError as error:
            messages = {
                429: "Gemini's free quota is exhausted or rate-limited. Try later. No paid fallback or automatic retry was used.",
                401: "The Gemini API key was rejected. Ask the app owner to check Streamlit Secrets.",
                403: "Gemini access was denied. Ask the app owner to check key permissions and regional availability.",
                404: "The configured Gemini model is unavailable. Ask the app owner to select an available free-tier model.",
            }
            message = messages.get(error.code, "The AI service is temporarily unavailable. Please try later.")
            error.close()
            raise AIServiceError(message) from None
        except (URLError, TimeoutError, OSError):
            raise AIServiceError("Cannot reach the AI service. Please try again later.") from None
        except (json.JSONDecodeError, UnicodeError, KeyError, TypeError, AttributeError, IndexError):
            raise AIServiceError("The AI service returned an unreadable response. Please try again.") from None
        finally:
            _gate.release()

    def invoke(self, prompt):
        response = self._run(prompt, False)
        try:
            return next(response)
        finally:
            response.close()

    def stream(self, prompt):
        yield from self._run(prompt, True)


def make_chat_model(local_factory, **kwargs):
    if ai_provider() == "gemini":
        return GeminiChat(**kwargs)
    return local_factory(**ollama_settings(), **kwargs)
