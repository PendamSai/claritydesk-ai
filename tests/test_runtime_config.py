import unittest
from unittest.mock import patch
from runtime_config import ollama_settings, public_deployment

class RuntimeConfigTests(unittest.TestCase):
    def test_local_default_preserved(self):
        with patch.dict("os.environ", {"OLLAMA_BASE_URL":"", "PUBLIC_DEPLOYMENT":"false"}):
            self.assertEqual(ollama_settings(), {})
            self.assertFalse(public_deployment())

    def test_private_hostport_gets_http_scheme(self):
        with patch.dict("os.environ", {"OLLAMA_BASE_URL":"ollama:11434", "PUBLIC_DEPLOYMENT":"true"}):
            self.assertEqual(ollama_settings(), {"base_url":"http://ollama:11434"})
            self.assertTrue(public_deployment())

    def test_https_is_preserved(self):
        with patch.dict("os.environ", {"OLLAMA_BASE_URL":"https://models.example.org"}):
            self.assertEqual(ollama_settings()["base_url"], "https://models.example.org")

    def test_invalid_endpoint_is_rejected(self):
        with patch.dict("os.environ", {"OLLAMA_BASE_URL":"file:///tmp/model"}):
            with self.assertRaises(ValueError):
                ollama_settings()
