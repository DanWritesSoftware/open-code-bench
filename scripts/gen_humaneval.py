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
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ocb.gateway.client import GatewayClient
from ocb.provenance import dataset_hash, git_commit

# Load local-only config (.env, gitignored) so OLLAMA_PI_BASE resolves without hardcoding
# a host IP in the repo. python-dotenv ships with litellm; degrade gracefully if absent.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GATEWAY = "http://localhost:4000/v1"
MODEL = "qwen-pi-1.5b"          # gateway logical name (routes to Pi Ollama)
BENCHMARK = "humaneval_plus"
OLLAMA_DIRECT = os.environ.get("OLLAMA_PI_BASE", "http://pi.lan:11434")   # backend provenance only; from .env

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function. "
    "Respond with only the completed function in a single Python code block."
)


def build_user_prompt(problem: dict) -> str:
    return (
        "Complete the following Python function. Return the full function in a single "
        "```python code block.\n\n" + problem["prompt"]
    )


def backend_info(model: str) -> dict:
    """Best-effort backend provenance (D15). Pi/Ollama: query /api/show for the quant.
    Other backends (e.g. vLLM on the Spark): record a generic note — dtype/quant are set
    by the server's launch args, not queried here."""
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
        return {"server": "ollama", "ollama_model": "qwen2.5-coder:1.5b", "quantization_level": "Q4_K_M (assumed)", "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="gateway logical model name (e.g. qwen-pi-1.5b, qwen-spark-7b)")
    ap.add_argument("--limit", type=int, default=5, help="number of problems (0 = all 164)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window (avoid silent truncation)")
    ap.add_argument("--concurrency", type=int, default=1, help="concurrent in-flight requests (>1 lets vLLM batch; keep 1 for the Pi)")
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
        "model_logical": args.model,
        "backend": backend_info(args.model),
        "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "options": {"num_ctx": args.num_ctx},
        "system_prompt": SYSTEM_PROMPT,
        "gateway": GATEWAY,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"run_id={run_id}  model={args.model}  ->  {out}")
    print(f"backend: {manifest['backend']}")
    print(f"generating {len(task_ids)} task(s) @ temp={args.temperature}, "
          f"max_tokens={args.max_tokens}, num_ctx={args.num_ctx}\n")

    client = GatewayClient(GATEWAY)
    records_f = (out / "records.jsonl").open("w", encoding="utf-8")
    samples_f = (out / "samples.jsonl").open("w", encoding="utf-8")

    n_ok = n_len = n_err = 0
    write_lock = threading.Lock()

    def gen_one(tid):
        """Generate one solution and thread-safely write its record + sample. Returns gen_status.
        Writes are serialized; completion-order in the files is fine — scoring keys by task_id and
        each task has a single sample (sample_index 0)."""
        nonlocal n_ok, n_len, n_err
        t0 = time.time()
        try:
            comp = client.complete(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(problems[tid])},
                ],
                model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                run_id=run_id, benchmark=BENCHMARK, task_id=tid, sample_index=0,
                num_ctx=args.num_ctx,
            )
            gen_status = "truncated" if comp.finish_reason == "length" else "ok"  # length == cut off
            rec = {
                "run_id": run_id, "task_id": tid, "sample_index": 0,
                "gen_status": gen_status, "finish_reason": comp.finish_reason,
                "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens,
                "latency_s": comp.latency_s,
                "raw_completion": comp.content,
            }
            sample = {"task_id": tid, "solution": comp.content}   # raw; #3 runs evalplus.sanitize
        except Exception as e:
            gen_status = "infra_error"
            rec = {
                "run_id": run_id, "task_id": tid, "sample_index": 0,
                "gen_status": "infra_error", "error": repr(e), "latency_s": round(time.time() - t0, 2),
            }
            sample = None
        with write_lock:
            if gen_status == "ok":
                n_ok += 1
            elif gen_status == "truncated":
                n_len += 1
            else:
                n_err += 1
            records_f.write(json.dumps(rec) + "\n"); records_f.flush()
            if sample is not None:
                samples_f.write(json.dumps(sample) + "\n"); samples_f.flush()
            done = n_ok + n_len + n_err
            print(f"[{done}/{len(task_ids)}] {tid}: {gen_status}, "
                  f"{rec.get('completion_tokens', '-')} tok, {rec['latency_s']}s")
        return gen_status

    if args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(gen_one, task_ids))   # vLLM batches the in-flight requests
    else:
        for tid in task_ids:
            gen_one(tid)

    records_f.close()
    samples_f.close()
    print(f"\nDone. ok(stop)={n_ok}  truncated(length)={n_len}  infra_error={n_err}")
    print(f"Artifacts: {out}\\  (manifest.json, records.jsonl, samples.jsonl)")


if __name__ == "__main__":
    main()
