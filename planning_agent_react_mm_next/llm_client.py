"""OpenAI-compatible multimodal chat client."""

from __future__ import annotations

from dataclasses import dataclass
import json
from threading import Lock
from typing import Any, Dict, List, Optional

from openai import OpenAI


PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "vllm": "http://127.0.0.1:8000/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2200
    timeout: int = 120
    dump_api_io: bool = False
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None


class OpenAICompatibleClient:
    """Minimal chat client supporting text and image_url message items."""

    _usage_lock = Lock()
    _global_usage: Dict[str, int] = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        base = cfg.base_url or PROVIDER_BASE_URLS.get(cfg.provider)
        if not base:
            raise ValueError(
                f"Unknown provider `{cfg.provider}`. Use one of {sorted(PROVIDER_BASE_URLS.keys())} or set base_url."
            )
        self.base_url = base.rstrip("/")
        # vLLM OpenAI-compatible endpoint often runs without key auth.
        api_key = self.cfg.api_key
        if not api_key and str(self.cfg.provider).lower() == "vllm":
            api_key = "EMPTY"
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.cfg.timeout,
        )
        self._req_seq = 0

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._req_seq += 1
        req_id = self._req_seq
        if self.cfg.dump_api_io:
            print(f"[LLM_IO][{req_id}] INPUT_MESSAGES_BEGIN")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            print(f"[LLM_IO][{req_id}] INPUT_MESSAGES_END")
        base_payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        extra_body: Dict[str, Any] = {}
        if self.cfg.enable_thinking is not None:
            if "dashscope.aliyuncs.com" in self.base_url:
                extra_body["enable_thinking"] = bool(self.cfg.enable_thinking)
            else:
                extra_body["thinking"] = {"type": "enabled" if self.cfg.enable_thinking else "disabled"}
        if self.cfg.reasoning_effort:
            base_payload["reasoning_effort"] = str(self.cfg.reasoning_effort)
        if extra_body:
            base_payload["extra_body"] = extra_body
        try:
            resp = self._client.chat.completions.create(**base_payload)
            out = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            usage = self._extract_usage(out)
            if usage is not None:
                cumulative = self._accumulate_usage(usage)
                print(
                    "[LLM_USAGE]"
                    f"[{req_id}] prompt={usage['prompt_tokens']} "
                    f"completion={usage['completion_tokens']} "
                    f"total={usage['total_tokens']} "
                    f"cumulative_prompt={cumulative['prompt_tokens']} "
                    f"cumulative_completion={cumulative['completion_tokens']} "
                    f"cumulative_total={cumulative['total_tokens']} "
                    f"requests={cumulative['request_count']}"
                )
            if self.cfg.dump_api_io:
                print(f"[LLM_IO][{req_id}] OUTPUT_CONTENT_BEGIN")
                print(self.extract_text(out))
                print(f"[LLM_IO][{req_id}] OUTPUT_CONTENT_END")
            return out
        except Exception as e:
            # Compatibility fallback for newer model contracts (e.g. requiring max_completion_tokens).
            err_msg = str(e)
            payload2 = dict(base_payload)
            used_fallback = False
            if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
                payload2.pop("max_tokens", None)
                payload2["max_completion_tokens"] = self.cfg.max_tokens
                used_fallback = True
            if "Unsupported parameter" in err_msg and "'temperature'" in err_msg:
                payload2.pop("temperature", None)
                used_fallback = True
            if used_fallback:
                try:
                    resp2 = self._client.chat.completions.create(**payload2)
                    out2 = resp2.model_dump() if hasattr(resp2, "model_dump") else dict(resp2)
                    usage = self._extract_usage(out2)
                    if usage is not None:
                        cumulative = self._accumulate_usage(usage)
                        print(
                            "[LLM_USAGE]"
                            f"[{req_id}] prompt={usage['prompt_tokens']} "
                            f"completion={usage['completion_tokens']} "
                            f"total={usage['total_tokens']} "
                            f"cumulative_prompt={cumulative['prompt_tokens']} "
                            f"cumulative_completion={cumulative['completion_tokens']} "
                            f"cumulative_total={cumulative['total_tokens']} "
                            f"requests={cumulative['request_count']} "
                            f"(fallback=compat_params)"
                        )
                    if self.cfg.dump_api_io:
                        print(f"[LLM_IO][{req_id}] OUTPUT_CONTENT_BEGIN")
                        print(self.extract_text(out2))
                        print(f"[LLM_IO][{req_id}] OUTPUT_CONTENT_END")
                    return out2
                except Exception as e2:
                    raise RuntimeError(
                        f"LLM request failed after compatibility fallback: {e2}; original_error={e}"
                    ) from e2
            raise RuntimeError(f"LLM request failed: {e}") from e

    @staticmethod
    def extract_text(response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Invalid LLM response (no choices): {response}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if txt:
                        chunks.append(str(txt))
            return "\n".join(chunks)
        return str(content)

    @classmethod
    def reset_global_usage(cls) -> None:
        with cls._usage_lock:
            cls._global_usage = {
                "request_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

    @classmethod
    def get_global_usage(cls) -> Dict[str, int]:
        with cls._usage_lock:
            return dict(cls._global_usage)

    @classmethod
    def _accumulate_usage(cls, usage: Dict[str, int]) -> Dict[str, int]:
        with cls._usage_lock:
            cls._global_usage["request_count"] += 1
            cls._global_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            cls._global_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            cls._global_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            return dict(cls._global_usage)

    @staticmethod
    def _extract_usage(response: Dict[str, Any]) -> Optional[Dict[str, int]]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("input_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")
        if prompt_tokens is None and completion_tokens is None:
            return None
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        total = usage.get("total_tokens")
        if total is None:
            total = p + c
        return {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": int(total or 0),
        }
