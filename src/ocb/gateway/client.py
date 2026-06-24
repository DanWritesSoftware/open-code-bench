"""GatewayClient — the single path generation flows through (D1).

An OpenAI-compatible client to the LiteLLM gateway that tags every call with run provenance
(run_id, benchmark, task) and returns a normalized `Completion`. Concurrency is the caller's
concern (the runner, or a ThreadPool in a script) — this class is one request per `complete()`.

Extracted from scripts/gen_humaneval.py during the Phase-1 refactor; the `chat.completions.create`
call shape is unchanged (incl. the Ollama-only `num_ctx`, dropped for other providers via the
gateway's drop_params).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Completion:
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class GatewayClient:
    def __init__(self, base_url: str, api_key: str = "sk-noauth"):
        # No real auth: the gateway is open on 127.0.0.1 (see litellm/config.yaml).
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, messages, *, model: str, temperature: float, max_tokens: int,
                 run_id: str, benchmark: str, task_id: str, sample_index: int = 0,
                 num_ctx: int | None = None) -> Completion:
        extra_body: dict = {
            "metadata": {
                "run_id": run_id, "benchmark": benchmark,
                "task_id": task_id, "sample_index": sample_index, "model_logical": model,
            },
        }
        if num_ctx is not None:
            extra_body["num_ctx"] = num_ctx
        t0 = time.time()
        r = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            extra_body=extra_body,
        )
        dt = time.time() - t0
        ch = r.choices[0]
        return Completion(
            content=ch.message.content or "",
            finish_reason=ch.finish_reason,
            prompt_tokens=r.usage.prompt_tokens,
            completion_tokens=r.usage.completion_tokens,
            latency_s=round(dt, 2),
        )
