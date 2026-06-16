"""open-code-bench Phase 0 — HumanEval+ generation (no scoring yet).

Loads real HumanEval+ problems via EvalPlus and generates one solution per problem
THROUGH THE GATEWAY (OpenAI SDK -> http://localhost:4000), writing to runs/<run_id>/:

  manifest.json   provenance (D5/D15): model + backend quant, sampling/options,
                  system prompt, dataset version + hash, evalplus version, gateway,
                  git commit, timestamp.
  records.jsonl   one row per (task_id, sample_index): raw completion, finish_reason,
                  usage, latency, gen_status (D12: 'ok' | 'truncated' | 'infra_error').
  samples.jsonl   EvalPlus-format {task_id, solution} (raw) for scoring in step #3.

Validated settings baked in (Phase-0 review): temperature=0 (Ollama default is 0.8),
explicit recorded system prompt (else Ollama injects Qwen's default), generous
max_tokens with finish_reason captured, and num_ctx sent per request.

Scoring (sanitize fences + run tests) is step #3 and needs Docker.

Run:  .venv\\Scripts\\python.exe scripts\\gen_humaneval.py --limit 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

GATEWAY = "http://localhost:4000/v1"
MODEL = "qwen-pi-1.5b"          # gateway logical name (routes to Pi Ollama)
BENCHMARK = "humaneval_plus"
OLLAMA_DIRECT = "http://172.18.70.178:11434"   # for backend provenance only

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function. "
    "Respond with only the completed function in a single Python code block."
)


def build_user_prompt(problem: dict) -> str:
    return (
        "Complete the following Python function. Return the full function in a single "
        "```python code block.\n\n" + problem["prompt"]
    )


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def dataset_hash(problems: dict) -> str:
    # Hash the FULL problem (prompt + base_input + plus_input + test + entry_point + ...),
    # NOT just the prompt: the "+" in HumanEval+ is the EXTRA test inputs. A prompt-only
    # hash wouldn't change if EvalPlus revised plus_input -> false reproducibility.
    blob = json.dumps({k: problems[k] for k in sorted(problems)},
                      sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def backend_info() -> dict:
    """Best-effort: ask the Pi's Ollama for the model details (quant) — provenance (D15)."""
    try:
        import urllib.request
        req = urllib.request.Request(
            OLLAMA_DIRECT + "/api/show",
            data=json.dumps({"model": "qwen2.5-coder:1.5b"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.load(r)
        return {"ollama_model": "qwen2.5-coder:1.5b", "details": d.get("details")}
    except Exception as e:
        return {"ollama_model": "qwen2.5-coder:1.5b", "quantization_level": "Q4_K_M (assumed)", "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5, help="number of problems (0 = all 164)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window (avoid silent truncation)")
    args = ap.parse_args()

    from evalplus.data import get_human_eval_plus
    import importlib.metadata as md

    problems = get_human_eval_plus()
    task_ids = list(problems)
    if args.limit:
        task_ids = task_ids[: args.limit]

    run_id = "heplus_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "benchmark": BENCHMARK,
        "evalplus_version": md.version("evalplus"),
        "dataset_size_total": len(problems),
        "dataset_size_run": len(task_ids),
        "dataset_hash": dataset_hash(problems),   # full-problem hash (covers plus_input)
        "model_logical": MODEL,
        "backend": backend_info(),
        "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "options": {"num_ctx": args.num_ctx},
        "system_prompt": SYSTEM_PROMPT,
        "gateway": GATEWAY,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"run_id={run_id}  ->  {out}")
    print(f"backend: {manifest['backend']}")
    print(f"generating {len(task_ids)} task(s) @ temp={args.temperature}, "
          f"max_tokens={args.max_tokens}, num_ctx={args.num_ctx}\n")

    client = OpenAI(base_url=GATEWAY, api_key="sk-noauth")
    records_f = (out / "records.jsonl").open("w", encoding="utf-8")
    samples_f = (out / "samples.jsonl").open("w", encoding="utf-8")

    n_ok = n_len = n_err = 0
    for i, tid in enumerate(task_ids, 1):
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(problems[tid])},
                ],
                extra_body={
                    "num_ctx": args.num_ctx,
                    "metadata": {
                        "run_id": run_id, "benchmark": BENCHMARK,
                        "task_id": tid, "sample_index": 0, "model_logical": MODEL,
                    },
                },
            )
            dt = time.time() - t0
            ch = r.choices[0]
            content = ch.message.content or ""
            if ch.finish_reason == "length":
                n_len += 1
                gen_status = "truncated"   # cut off by max_tokens: NOT a clean answer.
            else:                          # #3 scoring must bucket this apart from a
                n_ok += 1                  # genuine wrong answer (D12 infra-vs-model).
                gen_status = "ok"
            rec = {
                "run_id": run_id, "task_id": tid, "sample_index": 0,
                "gen_status": gen_status, "finish_reason": ch.finish_reason,
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
                "latency_s": round(dt, 2),
                "raw_completion": content,
            }
            # EvalPlus reads {task_id, solution}; we store the RAW completion here and let
            # step #3 run evalplus.sanitize (strip ```python fences) before scoring.
            # Truncated solutions are still written, but flagged gen_status='truncated' in
            # records.jsonl so #3 buckets them apart from genuine wrong answers.
            samples_f.write(json.dumps({"task_id": tid, "solution": content}) + "\n")
            print(f"[{i}/{len(task_ids)}] {tid}: {ch.finish_reason}, "
                  f"{r.usage.completion_tokens} tok, {dt:.1f}s")
        except Exception as e:
            dt = time.time() - t0
            n_err += 1
            rec = {
                "run_id": run_id, "task_id": tid, "sample_index": 0,
                "gen_status": "infra_error", "error": repr(e), "latency_s": round(dt, 2),
            }
            print(f"[{i}/{len(task_ids)}] {tid}: INFRA ERROR -> {e}")
        records_f.write(json.dumps(rec) + "\n")
        records_f.flush()
        samples_f.flush()

    records_f.close()
    samples_f.close()
    print(f"\nDone. ok(stop)={n_ok}  truncated(length)={n_len}  infra_error={n_err}")
    print(f"Artifacts: {out}\\  (manifest.json, records.jsonl, samples.jsonl)")


if __name__ == "__main__":
    main()
