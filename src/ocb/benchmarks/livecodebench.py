"""LiveCodeBench benchmark plugin — the third plugin (ARCHITECTURE.md §6, the LiveCodeBench row).

New dimension vs HumanEval+/BigCodeBench (§6): **time-windowed dataset versions** (D15). The dataset
`livecodebench/code_generation_lite` ships cumulative release tags (release_v1..release_v6 /
release_latest) plus optional contest-date filtering; a run pins one `release_version` (and,
optionally, a start/end contest-date window) and records it in the manifest.

Two wrinkles this plugin has to handle that the earlier two didn't:

1. **The HF dataset is a trust_remote_code builder script**, which `datasets>=4.0` refuses to run
   ("Dataset scripts are no longer supported"). So *generation* (this workstation, datasets 5.x)
   does NOT call `load_dataset`: it downloads the release's `testN.jsonl` files straight from the
   Hub and reads the three plain fields it needs to build a prompt (question_content, starter_code,
   question_id). *Scoring* runs the official evaluator inside the sandbox image, which pins an older
   `datasets` and has the dataset cache baked (D11, offline).

2. **The evaluator consumes already-extracted code, not raw completions.** LiveCodeBench's
   `custom_evaluator` path feeds each entry's `code_list` verbatim to the executor (no fence
   stripping — unlike EvalPlus/BigCodeBench's sanitize step). So we extract the last ```-fenced code
   block ourselves at score time (matching lcb_runner's generic-chat-model `extract_code`), keeping
   the raw completion in records.jsonl for provenance (same gen/score split as the other plugins).

Prompt is the official generic-chat template (SYSTEM_MESSAGE_GENERIC + get_generic_question_template
_answer) so numbers are comparable to the public leaderboard's non-CoT chat entries.

Pinned (D15): lcb_runner @ 28fef95 (2025-07-15); dataset livecodebench/code_generation_lite.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ocb.benchmarks.base import Benchmark, Message, Metrics, Task

# The pinned lcb_runner commit baked into the ocb-lcb-exec image (docker/exec-lcb/Dockerfile).
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
HF_DATASET = "livecodebench/code_generation_lite"

# The official generic-chat prompt (lcb_runner/prompts/code_generation.py), replicated verbatim so
# generation matches the public evaluator's non-CoT chat template.
SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests."
)
FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your "
    "code within delimiters."
)
FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
    "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def _release_files(release_version: str) -> list[str]:
    """Map a release tag to its cumulative testN.jsonl list (matches the HF builder's config).

    release_vN = test.jsonl + test2.jsonl .. testN.jsonl; release_latest == release_v6."""
    rv = "release_v6" if release_version == "release_latest" else release_version
    if not rv.startswith("release_v") or not rv[len("release_v"):].isdigit():
        raise ValueError(f"unsupported release_version {release_version!r} "
                         f"(expected release_v1..release_v6 or release_latest)")
    n = int(rv[len("release_v"):])
    return ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, n + 1)]


def _extract_code(completion: str) -> str:
    """Last ```-fenced code block, matching lcb_runner's generic-chat `extract_code`."""
    lines = completion.split("\n")
    fences = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1: fences[-1]])


class LiveCodeBench(Benchmark):
    name = "livecodebench"
    conversation_mode = "single_shot"
    system_prompt = SYSTEM_MESSAGE_GENERIC
    sandbox_image = "ocb-lcb-exec:0.1.0"   # offline evaluator image (docker/exec-lcb, D11)
    run_prefix = "lcb"
    sandbox_read_only = False    # the datasets lib + the evaluator write lock/temp files at runtime
    sandbox_auto_confirm = False

    def __init__(self, release_version: str = "release_v5",
                 start_date: str | None = None, end_date: str | None = None):
        self.release_version = release_version
        self.start_date = start_date       # YYYY-MM-DD contest-date window (applied on BOTH sides)
        self.end_date = end_date
        self._files = _release_files(release_version)   # validates the tag
        self._problems: list[dict] | None = None
        win = ""
        if start_date or end_date:
            win = f"-{start_date or 'min'}_{end_date or 'max'}"
        self.version = f"LiveCodeBench-lite-{release_version}{win}"

    # ---- dataset (generation side: read the Hub jsonl directly; no trust_remote_code) ----
    def _load(self) -> list[dict]:
        if self._problems is None:
            from datetime import datetime
            from huggingface_hub import hf_hub_download
            rows: list[dict] = []
            for fn in self._files:
                path = hf_hub_download(HF_DATASET, fn, repo_type="dataset")
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
            # apply the same contest-date window the evaluator will (lcb load_code_generation_dataset)
            if self.start_date:
                d0 = datetime.strptime(self.start_date, "%Y-%m-%d")
                rows = [r for r in rows if datetime.fromisoformat(r["contest_date"]) >= d0]
            if self.end_date:
                d1 = datetime.strptime(self.end_date, "%Y-%m-%d")
                rows = [r for r in rows if datetime.fromisoformat(r["contest_date"]) <= d1]
            rows.sort(key=lambda r: r["question_id"])   # match the evaluator's ordering
            self._problems = rows
        return self._problems

    def load_dataset(self, limit: int = 0) -> list[Task]:
        rows = self._load()
        if limit:
            rows = rows[:limit]
        return [Task(task_id=r["question_id"], data=r) for r in rows]

    def build_prompt(self, task: Task) -> list[Message]:
        q = task.data
        prompt = f"### Question:\n{q['question_content']}\n\n"
        if q.get("starter_code"):
            prompt += f"### Format: {FORMATTING_WITH_STARTER}\n"
            prompt += f"```python\n{q['starter_code']}\n```\n\n"
        else:
            prompt += f"### Format: {FORMATTING_WITHOUT_STARTER}\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        return [
            {"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": prompt},
        ]

    # extract_solution: inherit the default (raw completion). We extract the fenced code block at
    # score time (LiveCodeBench feeds code_list verbatim to the executor), mirroring the BCB plugin.

    def dataset_hash(self) -> str:
        rows = self._load()
        blob = json.dumps(
            [{"question_id": r["question_id"], "question_content": r["question_content"],
              "starter_code": r.get("starter_code", "")} for r in rows],
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()

    def backend_info(self, model: str) -> dict:
        if model.startswith("qwen-pi"):
            return {"server": "ollama", "model_logical": model}
        return {"server": "vllm", "served_model_logical": model,
                "note": "served via vLLM; dtype/quant set at vLLM launch"}

    # ---- scoring seam (delegated to lcb_runner's evaluator inside the sandbox) ----
    def evaluate(self, run_dir, *, sandbox=None, timeout: int = 1800, skip_eval: bool = False,
                 num_process_evaluate: int = 8, per_test_timeout: int = 6, parallel: int | None = None,
                 **_):
        # `parallel` (the runner's generic --parallel CLI flag) overrides num_process_evaluate when
        # given — lets a memory-constrained sandbox host throttle the evaluator's worker count
        # without touching the benchmark's default.
        if parallel:
            num_process_evaluate = parallel
        run_dir = Path(run_dir)
        custom_name = "lcb_custom.json"
        eval_all_name = "lcb_custom_codegeneration_output_eval_all.json"
        if not skip_eval:
            # 1. build the evaluator's input: one {question_id, code_list:[extracted_code]} per record
            #    (limit=0 generation covers every question_id, so len == len(benchmark) UNLESS a date
            #    window is set, in which case the wrapper's benchmark is filtered down (D15 contest-
            #    date windowing) and custom_outputs must match that windowed count exactly — the
            #    wrapper asserts len(custom_outputs) == len(benchmark)).
            records = self._read_records(run_dir)
            if self.start_date or self.end_date:
                windowed_qids = {str(r["question_id"]) for r in self._load()}
                records = [r for r in records if str(r["task_id"]) in windowed_qids]
            custom = [{"question_id": r["task_id"],
                       "code_list": [_extract_code(r.get("raw_completion", "") or "")
                                     if r.get("gen_status") != "infra_error" else ""]}
                      for r in records]
            (run_dir / custom_name).write_text(json.dumps(custom), encoding="utf-8")
            # 2. run the evaluator in the sandbox. The ocb-lcb-exec ENTRYPOINT is our thin wrapper
            #    (avoids lcb's torch/vendor-SDK generation deps); inner_cmd is just its args.
            inner = ["--custom_output_file", f"/work/{custom_name}",
                     "--release_version", self.release_version,
                     "--num_process_evaluate", str(num_process_evaluate),
                     "--timeout", str(per_test_timeout)]
            if self.start_date:
                inner += ["--start_date", self.start_date]
            if self.end_date:
                inner += ["--end_date", self.end_date]
            if sandbox is None:
                raise ValueError("evaluate() needs a SandboxRunner unless skip_eval=True")
            if sandbox.local:
                sandbox.run_local(run_dir, inner)
            else:
                sandbox.run_ssh(run_dir, inner, in_files=[custom_name], out_files=[eval_all_name])
            if sandbox.dry_run:
                print("[dry-run] sandbox not executed; skipping merge/score.")
                return Metrics(summary={"dry_run": True})
        return self._merge_and_score(run_dir, eval_all_name)

    @staticmethod
    def _read_records(run_dir: Path) -> list[dict]:
        return [json.loads(l) for l in
                (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def _merge_and_score(self, run_dir: Path, eval_all_name: str) -> Metrics:
        """Join lcb eval results onto records.jsonl -> D12 eval_status + pass@1.
        eval_all is a list of dicts each with question_id + graded_list (bool per sample)."""
        eval_all = json.loads((run_dir / eval_all_name).read_text(encoding="utf-8"))
        passed_by_qid: dict[str, bool] = {}
        for e in eval_all:
            gl = e.get("graded_list") or []
            passed_by_qid[str(e["question_id"])] = bool(gl[0]) if gl else False

        records = self._read_records(run_dir)
        n_ok = n_trunc = n_infra = n_scored = n_passed = n_fair = 0
        rows = []
        for rec in records:
            tid, gs = str(rec["task_id"]), rec.get("gen_status")
            if gs == "infra_error":
                n_infra += 1
                rows.append({**rec, "eval_status": None, "passed": None}); continue
            n_trunc += gs == "truncated"; n_ok += gs == "ok"
            if tid not in passed_by_qid:
                rows.append({**rec, "eval_status": "sandbox_error", "passed": None}); continue
            passed = passed_by_qid[tid]
            n_scored += 1
            if gs == "ok":
                n_fair += 1; n_passed += passed
            rows.append({**rec, "eval_status": "passed" if passed else "failed", "passed": passed})

        (run_dir / "scores.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        summary = {
            "run_id": run_dir.name, "benchmark": self.name, "version": self.version,
            "release_version": self.release_version,
            "start_date": self.start_date, "end_date": self.end_date,
            "total_records": len(records), "gen_ok": n_ok, "gen_truncated": n_trunc,
            "gen_infra_error": n_infra, "scored": n_scored, "fairly_attempted": n_fair,
            "passed": n_passed, "pass@1": (n_passed / n_fair) if n_fair else None,
            "completeness": f"{n_scored}/{len(records)} scored, {n_infra} infra, {n_trunc} truncated",
        }
        (run_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return Metrics(summary=summary)
