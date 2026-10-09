import json
from http.client import IncompleteRead
import unittest
from unittest.mock import patch

from wm.models import DeepSeekBackend, GenerationConfig


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "look"}}]}
        ).encode("utf-8")


class DeepSeekBackendTest(unittest.TestCase):
    @patch("wm.models.deepseek_backend.urlopen", return_value=_FakeResponse())
    def test_generate_uses_chat_completion(self, mock_urlopen) -> None:
        backend = DeepSeekBackend(
            model_id="deepseek-chat",
            api_key="test-key",
            base_url="https://example.invalid",
            max_retries=0,
        )

        result = backend.generate(
            [{"role": "user", "content": "Return look"}],
            GenerationConfig(max_new_tokens=4, temperature=0.0),
        )

        self.assertEqual(result, "look")
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["max_tokens"], 4)
        self.assertEqual(payload["temperature"], 0.0)

    @patch("wm.models.deepseek_backend.urlopen", return_value=_FakeResponse())
    def test_reasoner_uses_final_content_and_omits_temperature(
        self, mock_urlopen
    ) -> None:
        backend = DeepSeekBackend(
            model_id="deepseek-reasoner",
            api_key="test-key",
            base_url="https://example.invalid",
            max_retries=0,
        )

        result = backend.generate(
            [{"role": "user", "content": "Return look"}],
            GenerationConfig(max_new_tokens=2048, temperature=0.0),
        )

        self.assertEqual(result, "look")
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-reasoner")
        self.assertNotIn("temperature", payload)

    @patch(
        "wm.models.deepseek_backend.urlopen",
        side_effect=[IncompleteRead(b""), _FakeResponse()],
    )
    def test_retries_incomplete_response(self, mock_urlopen) -> None:
        backend = DeepSeekBackend(
            model_id="deepseek-chat",
            api_key="test-key",
            base_url="https://example.invalid",
            max_retries=1,
        )
        result = backend.generate(
            [{"role": "user", "content": "Return look"}],
            GenerationConfig(max_new_tokens=4, temperature=0.0),
        )
        self.assertEqual(result, "look")
        self.assertEqual(mock_urlopen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
