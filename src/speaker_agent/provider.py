from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Protocol


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        ...


class GeminiProvider:
    """Gemini client through Google's OpenAI-compatible API."""

    def __init__(self, model: str, timeout: float, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout,
        )
        self.model = model

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            raise ProviderError(str(exc)) from exc


class OpenAIProvider:
    """OpenAI chat-completions provider with native function calling."""

    def __init__(self, model: str, timeout: float, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            raise ProviderError(str(exc)) from exc


class OllamaProvider:
    """Local Ollama chat provider with the same tool-call response shape."""

    def __init__(self, model: str, timeout: float, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        output_schema = {
            "type": "object",
            "required": ["answer_id", "claims"],
            "properties": {
                "answer_id": {"type": "string"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["quote", "start_char", "end_char", "label", "fact_ids", "reason"],
                        "properties": {
                            "quote": {"type": "string"},
                            "start_char": {"type": "integer"},
                            "end_char": {"type": "integer"},
                            "label": {"type": "string", "enum": ["supported", "contradicted", "unknown"]},
                            "fact_ids": {"type": "array", "items": {"type": "string"}},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "format": output_schema if any(message.get("role") == "tool" for message in messages) else "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Ollama request failed ({exc.code}): {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        return self._response_from_data(result)

    @staticmethod
    def _response_from_data(result: dict[str, Any]) -> Any:
        message_data = result.get("message")
        if not isinstance(message_data, dict):
            raise ProviderError("Ollama returned no message")
        tool_calls = []
        for index, call in enumerate(message_data.get("tool_calls") or []):
            function = call.get("function", {})
            tool_calls.append(
                SimpleNamespace(
                    id=f"ollama-call-{index}",
                    function=SimpleNamespace(
                        name=function.get("name"),
                        arguments=json.dumps(function.get("arguments", {})),
                    ),
                )
            )
        content = message_data.get("content")
        if not tool_calls and isinstance(content, str):
            try:
                requested = json.loads(content)
            except json.JSONDecodeError:
                requested = None
            if (
                isinstance(requested, dict)
                and requested.get("name") == "get_brand_facts"
                and isinstance(requested.get("arguments"), dict)
            ):
                tool_calls.append(
                    SimpleNamespace(
                        id="ollama-content-call-0",
                        function=SimpleNamespace(
                            name="get_brand_facts",
                            arguments=json.dumps(requested["arguments"]),
                        ),
                    )
                )
                content = None
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
