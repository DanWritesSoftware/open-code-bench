"""The Benchmark extension seam (ARCHITECTURE.md §6).

A benchmark owns *building prompts* and *extracting solutions*, and delegates *scoring* to its
native evaluator (D1). The runner owns the model-interaction loop and calls `run()` k× per task
(D8). `run()` defaults to single-shot; multi-turn/agentic benchmarks override it (D14).
Generation flows through the GatewayClient (D1); scoring happens in the SandboxRunner (D11).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ocb.gateway.client import GatewayClient

Message = dict  # {"role": str, "content": str}


@dataclass
class Task:
    task_id: str
    data: dict                       # benchmark-specific problem payload


@dataclass
class Solution:
    task_id: str
    sample_index: int
    text: str                        # the solution (raw completion for HumanEval+)
    gen_status: str                  # "ok" | "truncated" | "infra_error" (D12)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    error: str | None = None


@dataclass
class Metrics:
    summary: dict                    # e.g. {"pass@1": ..., "completeness": ...}


class Benchmark(ABC):
    name: str
    version: str
    conversation_mode: Literal["single_shot", "multi_turn"] = "single_shot"
    system_prompt: str | None = None

    @abstractmethod
    def load_dataset(self, limit: int = 0) -> list[Task]: ...

    @abstractmethod
    def build_prompt(self, task: Task) -> list[Message]: ...

    def extract_solution(self, completion: str, task: Task) -> str:
        """Default: the completion IS the solution. (HumanEval+ keeps the raw completion here and
        lets evalplus.sanitize strip ```python fences in bulk at score time.)"""
        return completion

    def run(self, task: Task, client: GatewayClient, *, model: str, sampling: dict,
            run_id: str, sample_index: int = 0) -> Solution:
        """Default single-shot driver (D14). Multi-turn benchmarks override this."""
        t0 = time.time()
        try:
            comp = client.complete(
                self.build_prompt(task),
                model=model, temperature=sampling["temperature"], max_tokens=sampling["max_tokens"],
                run_id=run_id, benchmark=self.name, task_id=task.task_id, sample_index=sample_index,
                num_ctx=sampling.get("num_ctx"),
            )
            gen_status = "truncated" if comp.finish_reason == "length" else "ok"
            return Solution(
                task_id=task.task_id, sample_index=sample_index,
                text=self.extract_solution(comp.content, task), gen_status=gen_status,
                finish_reason=comp.finish_reason, prompt_tokens=comp.prompt_tokens,
                completion_tokens=comp.completion_tokens, latency_s=comp.latency_s,
            )
        except Exception as e:
            return Solution(task_id=task.task_id, sample_index=sample_index, text="",
                            gen_status="infra_error", error=repr(e),
                            latency_s=round(time.time() - t0, 2))

    @abstractmethod
    def evaluate(self, run_dir, **opts) -> Metrics:
        """Score a generation run dir: native evaluator in the sandbox -> merge -> pass@k (D1/D11/D12)."""

    # Provenance hooks (D5/D15) — overridden by plugins that can compute them.
    def dataset_version(self) -> str:
        return self.version

    def dataset_hash(self) -> str | None:
        return None

    def backend_info(self, model: str) -> dict:
        return {"model_logical": model}
