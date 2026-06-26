"""HumanEval+ benchmark plugin — the first plugin (ARCHITECTURE.md §6).

Ports the HumanEval+ logic that used to live in scripts/gen_humaneval.py (prompts, backend
provenance) and scripts/score_humaneval.py (sanitize -> evalplus.evaluate in the sandbox ->
merge -> D12 eval_status -> pass@1 + completeness). Scoring is delegated to EvalPlus (D1) run
inside the hardened sandbox (D11). This is the single source of truth for HumanEval+; the
scripts are thin wrappers over it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ocb.benchmarks.base import Benchmark, Message, Metrics, Task
from ocb.provenance import dataset_hash as _dataset_hash

SANITIZED_NAME = "samples-sanitized.jsonl"
RESULT_NAME = "samples-sanitized_eval_results.json"   # evalplus: <samples>.jsonl -> <samples>_eval_results.json
PASS = "pass"
TIMEOUT = "timeout"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function. "
    "Respond with only the completed function in a single Python code block."
)
OLLAMA_DIRECT = os.environ.get("OLLAMA_PI_BASE", "http://pi.lan:11434")  # backend provenance only


class HumanEvalPlus(Benchmark):
    name = "humaneval_plus"
    version = "HumanEvalPlus-v0.1.10"          # the version evalplus 0.3.1 pins (D15)
    conversation_mode = "single_shot"
    system_prompt = SYSTEM_PROMPT
    run_prefix = "heplus"

    def __init__(self):
        self._problems = None

    def _load_problems(self) -> dict:
        if self._problems is None:
            from evalplus.data import get_human_eval_plus
            self._problems = get_human_eval_plus()
        return self._problems

    # ---- generation seam ----
    def load_dataset(self, limit: int = 0) -> list[Task]:
        problems = self._load_problems()
        tids = list(problems)
        if limit:
            tids = tids[:limit]
        return [Task(task_id=t, data=problems[t]) for t in tids]

    def build_prompt(self, task: Task) -> list[Message]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                "Complete the following Python function. Return the full function in a single "
                "```python code block.\n\n" + task.data["prompt"]},
        ]

    # extract_solution: inherit the default (raw completion); evalplus.sanitize strips fences
    # in bulk at score time, matching the original gen/score split.

    def dataset_hash(self) -> str:
        return _dataset_hash(self._load_problems())

    def backend_info(self, model: str) -> dict:
        """Pi/Ollama: query /api/show for the quant. vLLM/other: a generic note (D15)."""
        if not model.startswith("qwen-pi"):
            return {"server": "vllm", "served_model_logical": model,
                    "note": "served via vLLM; dtype/quant set at vLLM launch"}
        try:
            import urllib.request
            req = urllib.request.Request(
                OLLAMA_DIRECT + "/api/show",
                data=json.dumps({"model": "qwen2.5-coder:1.5b"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.load(r)
            return {"server": "ollama", "ollama_model": "qwen2.5-coder:1.5b", "details": d.get("details")}
        except Exception as e:
            return {"server": "ollama", "ollama_model": "qwen2.5-coder:1.5b",
                    "quantization_level": "Q4_K_M (assumed)", "error": str(e)}

    # ---- scoring seam (delegated to EvalPlus, run in the sandbox) ----
    @staticmethod
    def _eval_status_for(base_status, plus_status, base_only: bool):
        """Map EvalPlus (base_status, plus_status) onto a D12 (eval_status, passed) pair.
        EvalPlus folds compile-errors/raises into "fail" (no failed-vs-error split — deferred)."""
        statuses = [base_status] if base_only else [base_status, plus_status]
        if any(s == TIMEOUT for s in statuses):
            return "timeout", False
        if all(s == PASS for s in statuses):
            return "passed", True
        return "failed", False

    def _sanitize(self, run_dir: Path) -> None:
        samples = run_dir / "samples.jsonl"
        if not samples.is_file():
            sys.exit(f"error: {samples} not found (generate first)")
        subprocess.run([sys.executable, "-m", "evalplus.sanitize", str(samples)], check=True)

    def evaluate(self, run_dir, *, sandbox=None, dataset: str = "humaneval", base_only: bool = False,
                 timeout: int = 1800, parallel: int | None = None, skip_eval: bool = False) -> Metrics:
        run_dir = Path(run_dir)
        if not skip_eval:
            self._sanitize(run_dir)
            inner = ["timeout", f"{timeout}s", "python", "-m", "evalplus.evaluate", dataset,
                     "--samples", f"/work/{SANITIZED_NAME}"]
            if parallel:
                inner += ["--parallel", str(parallel)]
            if base_only:
                inner.append("--base_only")
            if sandbox is None:
                raise ValueError("evaluate() needs a SandboxRunner unless skip_eval=True")
            if sandbox.local:
                sandbox.run_local(run_dir, inner)
            else:
                sandbox.run_ssh(run_dir, inner, in_files=[SANITIZED_NAME], out_files=[RESULT_NAME])
            if sandbox.dry_run:
                print("[dry-run] sandbox not executed; skipping merge/score.")
                return Metrics(summary={"dry_run": True})
        return self._merge_and_score(run_dir, dataset, base_only)

    def _merge_and_score(self, run_dir: Path, dataset: str, base_only: bool) -> Metrics:
        """Join EvalPlus results onto records.jsonl, assign D12 eval_status, compute pass@1."""
        results = json.loads((run_dir / RESULT_NAME).read_text(encoding="utf-8"))
        eval_by_task = results.get("eval", {})
        records = [json.loads(line) for line in
                   (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        n_ok = n_trunc = n_infra = 0
        n_scored = n_passed = n_fair = 0
        rows = []
        for rec in records:
            tid, idx = rec["task_id"], rec.get("sample_index", 0)
            gen_status = rec.get("gen_status")
            if gen_status == "infra_error":
                n_infra += 1
                rows.append({**rec, "eval_status": None, "passed": None})
                continue
            if gen_status == "truncated":
                n_trunc += 1
            elif gen_status == "ok":
                n_ok += 1
            task_entries = eval_by_task.get(tid, [])
            if idx >= len(task_entries):
                rows.append({**rec, "eval_status": "sandbox_error", "passed": None})
                continue
            entry = task_entries[idx]
            eval_status, passed = self._eval_status_for(entry.get("base_status"),
                                                        entry.get("plus_status"), base_only)
            n_scored += 1
            if gen_status == "ok":                 # pass@1 over fairly-attempted samples (D12)
                n_fair += 1
                if passed:
                    n_passed += 1
            rows.append({**rec, "eval_status": eval_status, "passed": passed,
                         "base_status": entry.get("base_status"),
                         "plus_status": entry.get("plus_status")})

        (run_dir / "scores.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        pass_at_1 = (n_passed / n_fair) if n_fair else None
        summary = {
            "run_id": run_dir.name,
            "dataset": dataset + ("" if base_only else "+ (base+extra tests)"),
            "evalplus_dataset_hash": results.get("hash"),
            "total_records": len(records),
            "gen_ok": n_ok, "gen_truncated": n_trunc, "gen_infra_error": n_infra,
            "scored": n_scored, "fairly_attempted": n_fair, "passed": n_passed,
            "pass@1": pass_at_1,
            "completeness": f"{n_scored}/{len(records)} scored, {n_infra} infra, {n_trunc} truncated",
        }
        (run_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return Metrics(summary=summary)
