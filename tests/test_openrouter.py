"""Tests for OpenRouter LLM provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gator.llm_providers import call_llm, call_openrouter


class TestCallOpenrouter:
    """Test call_openrouter response parsing."""

    def _mock_response(self, content: str, status: int = 200) -> MagicMock:
        """Build a mock urllib response."""
        body = json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "model": "anthropic/claude-sonnet-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        ).encode("utf-8")
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    @patch("urllib.request.urlopen")
    def test_basic_response(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response("Hello from Claude!")
        result = call_openrouter("test prompt", model="anthropic/claude-sonnet-4")
        assert result == "Hello from Claude!"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    @patch("urllib.request.urlopen")
    def test_whitespace_stripped(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response("  response with spaces  \n")
        result = call_openrouter("test", model="test/model")
        assert result == "response with spaces"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    @patch("urllib.request.urlopen")
    def test_request_body_format(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response("ok")
        call_openrouter("my prompt", model="openai/gpt-4.1", temperature=0.5)

        # Verify the request was made with correct body
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "openai/gpt-4.1"
        assert body["messages"] == [{"role": "user", "content": "my prompt"}]
        assert body["temperature"] == 0.5

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    @patch("urllib.request.urlopen")
    def test_auth_header(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response("ok")
        call_openrouter("test", model="test/model")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer test-key-123"

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
            call_openrouter("test")

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    @patch("urllib.request.urlopen")
    def test_empty_choices_returns_raw_json(self, mock_urlopen):
        body = json.dumps({"choices": []}).encode("utf-8")
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = call_openrouter("test", model="test/model")
        # Should return raw JSON when no content extracted
        parsed = json.loads(result)
        assert parsed["choices"] == []


class TestCallLlmDispatcher:
    """Test that call_llm correctly dispatches to openrouter."""

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    @patch("gator.llm_providers.call_openrouter")
    def test_dispatch_openrouter(self, mock_or):
        mock_or.return_value = "dispatched ok"
        result = call_llm("openrouter", "test prompt", model="test/model")
        assert result == "dispatched ok"
        mock_or.assert_called_once()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            call_llm("unknown_provider", "test")
