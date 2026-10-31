import io
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
import ai_client
from ai_client import AIServiceError, GeminiChat, make_chat_model

class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AI_PROVIDER":"gemini", "GEMINI_API_KEY":"test-key-never-real", "GEMINI_DAILY_REQUEST_LIMIT":"100"})
        self.env.start()
        self.addCleanup(self.env.stop)
        with ai_client._usage_lock:
            ai_client._usage_day = ""
            ai_client._usage_count = 0

    def payload(self, text):
        return {"candidates":[{"content":{"parts":[{"text":text}]}}]}

    def test_invoke_uses_fixed_google_endpoint_and_header_key(self):
        with patch("ai_client.urlopen", return_value=io.BytesIO(json.dumps(self.payload("Answer")).encode())) as network:
            result = GeminiChat("gemini-2.5-flash-lite").invoke("Question")
        self.assertEqual(result.content, "Answer")
        request = network.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://generativelanguage.googleapis.com/"))
        self.assertNotIn("test-key-never-real", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key-never-real")
        self.assertNotIn("tools", json.loads(request.data))

    def test_streaming_combines_chunks_and_ignores_thoughts(self):
        stream = "data: " + json.dumps(self.payload("Hello ")) + "\n\ndata: " + json.dumps(self.payload("world")) + "\n\n"
        with patch("ai_client.urlopen", return_value=io.BytesIO(stream.encode())):
            self.assertEqual("".join(c.content for c in GeminiChat("model").stream("Q")), "Hello world")

    def test_json_mode_and_output_cap(self):
        model = GeminiChat("model", format="json", num_predict=9000)
        self.assertEqual(model.generation["responseMimeType"], "application/json")
        self.assertEqual(model.generation["maxOutputTokens"], 1500)

    def test_quota_failure_has_no_retry_and_no_secret_leak(self):
        with patch("ai_client.urlopen", side_effect=HTTPError("url",429,"rate limit",{},None)) as network:
            with self.assertRaises(AIServiceError) as raised:
                GeminiChat("model").invoke("Question")
        self.assertEqual(network.call_count, 1)
        self.assertIn("free quota", str(raised.exception))
        self.assertNotIn("test-key-never-real", str(raised.exception))

    def test_application_cap_blocks_before_network(self):
        with patch.dict(os.environ, {"GEMINI_DAILY_REQUEST_LIMIT":"1"}):
            with patch("ai_client.urlopen", return_value=io.BytesIO(json.dumps(self.payload("Answer")).encode())) as network:
                model = GeminiChat("model")
                model.invoke("First")
                with self.assertRaisesRegex(AIServiceError, "daily"):
                    model.invoke("Second")
                self.assertEqual(network.call_count, 1)

    def test_missing_key_blocks_without_fallback(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY":""}):
            local = Mock()
            with self.assertRaisesRegex(AIServiceError, "not configured"):
                make_chat_model(local, model="model")
            local.assert_not_called()

    def test_blocked_or_empty_output_is_clear_error(self):
        with patch("ai_client.urlopen", return_value=io.BytesIO(b'{"candidates": []}')):
            with self.assertRaisesRegex(AIServiceError, "no answer"):
                GeminiChat("model").invoke("Question")

    def test_local_factory_remains_available(self):
        with patch.dict(os.environ, {"AI_PROVIDER":"ollama", "OLLAMA_BASE_URL":""}):
            local = Mock()
            make_chat_model(local, model="llama3.2", num_predict=256)
            local.assert_called_once_with(model="llama3.2", num_predict=256)

    def test_free_mode_blocks_semantic_embedding_server(self):
        from rag import build_vector_store
        with patch("rag.OllamaEmbeddings") as embeddings:
            with self.assertRaisesRegex(ValueError, "keyword retrieval"):
                build_vector_store([("a.pdf", b"pdf")], "embedding")
            embeddings.assert_not_called()

    def test_free_pdf_page_limit_precedes_extraction(self):
        from rag import build_vector_store
        from types import SimpleNamespace
        with patch("rag.PdfReader", return_value=SimpleNamespace(pages=[object()] * 101)):
            with self.assertRaisesRegex(ValueError, "100 PDF pages"):
                build_vector_store([("a.pdf", b"pdf")], "unused", search_mode="keyword")

    def test_long_recording_is_rejected_before_transcription(self):
        import av
        from assistants import transcribe_audio
        with patch("av.open") as decoder, patch("assistants.speech_model") as model:
            decoder.return_value.__enter__.return_value.duration = 301 * av.time_base
            with self.assertRaisesRegex(ValueError, "5 minutes"):
                transcribe_audio(b"audio")
            model.assert_not_called()
