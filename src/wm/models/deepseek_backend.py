from __future__ import annotations

import json
import os
import time
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from ssl import SSLError
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import GenerationConfig, Message


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting shell variables."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _completion_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


class DeepSeekBackend:
    """DeepSeek backend using its OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        dotenv_path: str | Path | None = None,
    ) -> None:
        dotenv = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
        _load_dotenv(dotenv)

        # An explicit experiment config takes precedence over the optional
        # environment override. This keeps chat and reasoner experiments
        # reproducible when both are configured on the same machine.
        self.model_id = model_id or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Missing DeepSeek API key. Set {api_key_env} in .env or the shell."
            )

        configured_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
        self.url = _completion_url(
            configured_base_url or "https://api.deepseek.com"
        )
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def generate(
        self,
        messages: Sequence[Message],
        generation: GenerationConfig,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "max_tokens": generation.max_new_tokens,
            "top_p": generation.top_p,
            "stream": False,
        }
        # DeepSeek's reasoning model does not use temperature. Omitting it
        # also makes the request explicit and avoids relying on compatibility
        # behavior at the API boundary.
        if not self.model_id.endswith("-reasoner"):
            payload["temperature"] = generation.temperature
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                result = json.loads(body)
                return self._extract_content(result)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                # 520 is commonly returned by the proxy/API edge when the
                # origin resets a request.  It is transient in the same way
                # as the other gateway/server errors and must not abort a
                # long-running evaluation whose checkpoint is otherwise
                # healthy.
                if exc.code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                    520,
                    521,
                    522,
                    524,
                } or attempt == self.max_retries:
                    raise RuntimeError(
                        f"DeepSeek API returned HTTP {exc.code}: {body}"
                    ) from exc
                last_error = RuntimeError(
                    f"DeepSeek API returned retryable HTTP {exc.code}: {body}"
                )
            except (
                IncompleteRead,
                RemoteDisconnected,
                SSLError,
                URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(min(30.0, 2**attempt))
        raise RuntimeError(
            f"DeepSeek API request failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        ) from last_error

    @staticmethod
    def _extract_content(result: dict[str, Any]) -> str:
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "DeepSeek API response did not contain choices[0].message.content"
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek API returned non-text message content")
        return content.strip()
