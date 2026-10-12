"""Deterministic mock chat client for local smoke tests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class SequentialMockClient:
    """OpenAI-compatible mock client that replays predefined texts."""

    def __init__(self, outputs: Iterable[str]):
        self._outputs = list(outputs)
        self._index = 0

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._index >= len(self._outputs):
            raise RuntimeError(f"SequentialMockClient ran out of scripted outputs. Last feedback: {messages[-1].get('content')}")
        text = self._outputs[self._index]
        self._index += 1
        return {"choices": [{"message": {"content": text}}]}

    @staticmethod
    def extract_text(response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Invalid mock response: {response}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")
