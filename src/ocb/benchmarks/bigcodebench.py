"""BigCodeBench benchmark plugin — the second plugin (proves the framework seam, §6).

Same lineage as EvalPlus, so it mirrors HumanEval+: generate full functions through the gateway,
then score with `bigcodebench.evaluate` inside a sandbox that has the ~139 task libraries (D1/D11).
Differences from HumanEval+:
  * a FATTER sandbox image — the official `bigcodebench/bigcodebench-evaluate` (139 libs baked,
    runs offline with `--execution local`); its ENTRYPOINT is `bigcodebench.evaluate`, so the
    inner command we pass is just the args.
  * two axes: `split` (complete|instruct) and `subset` (full|hard), set per run via spec options.
  * a `bigcodebench.sanitize --calibrate` pre-step (local, pure text) before evaluation.
Pinned (D15): bigcodebench 0.2.5, dataset v0.1.4.

Eval path confirmed live (Qwen2.5-Coder-7B, hard/complete -> pass@1 0.224): sanitize emits
`samples-sanitized-calibrated.jsonl`; the evaluator writes `<stem>_eval_results.json` shaped
`{"date", "eval": {task_id: [{"task_id", "solution", "status": "pass"|..., "details"}]}}`. Two
sandbox quirks it needs (vs the strict `exec` profile): a writable rootfs (it caches a GT pickle)
and stdin auto-confirm (it prompts `[Y/N]` to overwrite results) — see base.py knobs.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

from ocb.benchmarks.base import Benchmark, Message, Metrics, Task

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Respond with only the complete function "
    "(including any imports it needs) in a single Python code block."
)


class BigCodeBench(Benchmark):
    name = "bigcodebench"
    conversation_mode = "single_shot"
    system_prompt = SYSTEM_PROMPT
    sandbox_image = "ocb-bcb-exec:0.2.4"   # our offline derivative (dataset baked, HF offline) of
    #                                        bigcodebench/bigcodebench-evaluate:v0.2.4 (D11 --network=none)
    run_prefix = "bcb"
    sandbox_read_only = False    # bigcodebench's evaluator writes a GT cache + per-task temp dirs
    sandbox_auto_confirm = True  # it prompts [Y/N] to overwrite results; auto-answer 'y' (no TTY over SSH)

    def __init__(self, split: str = "complete", subset: str = "full"):
        if split not in ("complete", "instruct"):
            raise ValueError("split must be 'complete' or 'instruct'")
        if subset not in ("full", "hard"):
            raise ValueError("subset must be 'full' or 'hard'")
        self.split = split
        self.subset = subset
        self._problems = None
        from bigcodebench.data.bigcodebench import BIGCODEBENCH_VERSION
        self.version = f"bigcodebench-{BIGCODEBENCH_VERSION}-{subset}-{split}"

    def _load(self) -> dict:
        if self._problems is None:
            from bigcodebench.data import get_bigcodebench
            self._problems = get_bigcodebench(subset=self.subset)
        return self._problems

    # ---- generation seam ----
    def load_dataset(self, limit: int = 0) -> list[Task]:
        probs = self._load()
        tids = list(probs)
        if limit:
            tids = tids[:limit]
        return [Task(task_id=t, data=probs[t]) for t in tids]

    def build_prompt(self, task: Task) -> list[Message]:
        if self.split == "instruct":
            user = task.data["instruct_prompt"]
        else:  # complete: hand the model the signature + docstring to fill in
            user = ("Complete this Python function — return the full function, including the "
                    "imports and signature, in a single ```python code block:\n\n"
                    + task.data["complete_prompt"])
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # extract_solution: inherit the default (raw completion); bigcodebench.sanitize --calibrate
    # extracts/assembles the final solution at score time, matching the gen/score split.

    def dataset_hash(self) -> str:
        from bigcodebench.data import get_bigcodebench_hash
        try:
            return get_bigcodebench_hash(subset=self.subset)
        except TypeError:
            return get_bigcodebench_hash()

    def backend_info(self, model: str) -> dict:
        if model.startswith("qwen-pi"):
            return {"server": "ollama", "model_logical": model}
        return {"server": "vllm", "served_model_logical": model,
                "note": "served via vLLM; dtype/quant set at vLLM launch"}

    # ---- scoring seam (delegated to bigcodebench.evaluate inside the sandbox) ----
    def _find_sanitized(self, run_dir: Path) -> Path:
        # sanitize emits a calibrated jsonl; the exact name varies by version, so glob for it.
        cands = sorted(glob.glob(str(run_dir / "samples*sanitized*calibrated*.jsonl"))) \
            or sorted(glob.glob(str(run_dir / "samples*sanitized*.jsonl")))
        if not cands:
            raise FileNotFoundError(f"no sanitized samples found in {run_dir} after bigcodebench.sanitize")
        return Path(cands[-1])

    def evaluate(self, run_dir, *, sandbox=None, timeout: int = 1800, skip_eval: bool = False, **_):
        run_dir = Path(run_dir)
        if not skip_eval:
            # 1. sanitize + calibrate (local, pure text — no task deps needed)
            subprocess.run([sys.executable, "-m", "bigcodebench.sanitize", "--calibrate",
                            "--samples", str(run_dir / "samples.jsonl")], check=True)
            sanitized = self._find_sanitized(run_dir)
            # 2. evaluate in the sandbox. The bigcodebench image's ENTRYPOINT is bigcodebench.evaluate,
            #    so inner_cmd is just its args. (timeout wrapper omitted: the image isn't guaranteed
            #    to have coreutils `timeout`; the SandboxRunner/host bounds the run instead.)
            inner = ["--split", self.split, "--subset", self.subset,
                     "--samples", f"/work/{sanitized.name}", "--execution", "local"]
            if sandbox is None:
                raise ValueError("evaluate() needs a SandboxRunner unless skip_eval=True")
            # Point bigcodebench at the dataset jsonl baked into ocb-bcb-exec — its get_bigcodebench
            # otherwise re-fetches from the HF Hub (fails under --network=none); the OVERRIDE_PATH
            # env is the documented air-gapped escape hatch.
            ds_jsonl = f"BigCodeBench-{'Hard-' if self.subset == 'hard' else ''}v0.1.4.jsonl"
            env = {"BIGCODEBENCH_OVERRIDE_PATH": f"/home/bigcodebenchuser/.cache/bigcodebench/{ds_jsonl}"}
            if sandbox.local:
                sandbox.run_local(run_dir, inner, env=env)
            else:
                sandbox.run_ssh(run_dir, inner, in_files=[sanitized.name],
                                out_files=[sanitized.stem + "_eval_results.json"], env=env)
            if sandbox.dry_run:
                print("[dry-run] sandbox not executed; skipping merge/score.")
                return Metrics(summary={"dry_run": True})
        return self._merge_and_score(run_dir)

    def _merge_and_score(self, run_dir: Path) -> Metrics:
        """Join bigcodebench eval results onto records.jsonl -> D12 eval_status + pass@1.
        eval_results.json shape (confirmed): {"eval": {task_id: [{"status": "pass"|...}]}}."""
        sanitized = self._find_sanitized(run_dir)
        results = json.loads((run_dir / (sanitized.stem + "_eval_results.json")).read_text(encoding="utf-8"))
        eval_by_task = results.get("eval", results)
        records = [json.loads(l) for l in
                   (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

        def status_of(entry):  # tolerate {"status": "pass"} or [..,"pass"] or nested shapes
            if isinstance(entry, dict):
                return entry.get("status") or entry.get("base_status")
            if isinstance(entry, (list, tuple)):
                return next((x for x in entry if isinstance(x, str)), None)
            return entry

        n_ok = n_trunc = n_infra = n_scored = n_passed = n_fair = 0
        rows = []
        for rec in records:
            tid, idx = rec["task_id"], rec.get("sample_index", 0)
            gs = rec.get("gen_status")
            if gs == "infra_error":
                n_infra += 1; rows.append({**rec, "eval_status": None, "passed": None}); continue
            n_trunc += gs == "truncated"; n_ok += gs == "ok"
            entries = eval_by_task.get(tid)
            if not entries:
                rows.append({**rec, "eval_status": "sandbox_error", "passed": None}); continue
            entry = entries[idx] if isinstance(entries, list) and idx < len(entries) else entries
            passed = status_of(entry) == "pass"
            n_scored += 1
            if gs == "ok":
                n_fair += 1; n_passed += passed
            rows.append({**rec, "eval_status": "passed" if passed else "failed", "passed": passed})

        (run_dir / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        summary = {
            "run_id": run_dir.name, "benchmark": self.name, "version": self.version,
            "split": self.split, "subset": self.subset,
            "total_records": len(records), "gen_ok": n_ok, "gen_truncated": n_trunc,
            "gen_infra_error": n_infra, "scored": n_scored, "fairly_attempted": n_fair,
            "passed": n_passed, "pass@1": (n_passed / n_fair) if n_fair else None,
            "completeness": f"{n_scored}/{len(records)} scored, {n_infra} infra, {n_trunc} truncated",
        }
        (run_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return Metrics(summary=summary)
