"""Spec-driven runner (ARCHITECTURE.md §5.6) — minimal.

Two operations, split to honor the Phase-0 network constraint (generate on the Pi/Spark network,
score on the Mint sandbox network); generation is also resumable-by-rerun since it writes per task:

  generate <spec.yaml>   for each model: run benchmark.run() per task via the gateway (optionally
                         concurrent, D8), writing runs/<run_id>/{manifest,records,samples}.jsonl
  score <run_dir>        benchmark.evaluate() in the hardened sandbox (D1/D11) -> scores + pass@k

CLI:
  python -m ocb.runner.run generate specs/heplus.yaml [--limit N] [--concurrency C]
  python -m ocb.runner.run score runs/<run_id> [--ssh-host H | --local] [--skip-eval] [--dry-run] ...
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ocb.benchmarks.registry import get_benchmark
from ocb.gateway.client import GatewayClient
from ocb.provenance import git_commit
from ocb.sandbox.runner import SandboxRunner

GATEWAY = "http://localhost:4000/v1"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate(spec: dict, *, limit: int | None = None, concurrency: int | None = None) -> list[str]:
    opts = dict(spec.get("options", {}))   # benchmark-specific (e.g. BigCodeBench split/subset)
    bench = get_benchmark(spec["benchmark"], **opts)
    all_tasks = bench.load_dataset(0)
    lim = limit if limit is not None else int(spec.get("limit", 0))
    tasks = all_tasks[:lim] if lim else all_tasks
    sampling = dict(spec["sampling"])
    conc = concurrency if concurrency is not None else int(spec.get("concurrency", 1))
    gateway = spec.get("gateway", GATEWAY)
    client = GatewayClient(gateway)
    models = list(spec["models"])
    ts = _now()
    run_ids = []
    for model in models:
        run_id = f"{bench.run_prefix}_{ts}" if len(models) == 1 else f"{bench.run_prefix}_{ts}_{model}"
        out = Path("runs") / run_id
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id, "benchmark": bench.name, "benchmark_version": bench.version,
            "benchmark_options": opts,
            "dataset_size_total": len(all_tasks), "dataset_size_run": len(tasks),
            "dataset_hash": bench.dataset_hash(),
            "model_logical": model, "backend": bench.backend_info(model),
            "sampling": sampling, "system_prompt": bench.system_prompt,
            "gateway": gateway, "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"run_id={run_id}  model={model}  ->  {out}  ({len(tasks)} tasks, concurrency={conc})")
        _generate_model(bench, client, tasks, model, sampling, run_id, out, conc)
        run_ids.append(run_id)
    return run_ids


def _generate_model(bench, client, tasks, model, sampling, run_id, out: Path, conc: int) -> None:
    records_f = (out / "records.jsonl").open("w", encoding="utf-8")
    samples_f = (out / "samples.jsonl").open("w", encoding="utf-8")
    lock = threading.Lock()
    counts: dict[str, int] = {"ok": 0, "truncated": 0, "infra_error": 0}

    def one(task):
        sol = bench.run(task, client, model=model, sampling=sampling, run_id=run_id)
        rec = {"run_id": run_id, "task_id": sol.task_id, "sample_index": sol.sample_index,
               "gen_status": sol.gen_status, "finish_reason": sol.finish_reason,
               "prompt_tokens": sol.prompt_tokens, "completion_tokens": sol.completion_tokens,
               "latency_s": sol.latency_s, "raw_completion": sol.text}
        if sol.error is not None:
            rec["error"] = sol.error
        with lock:
            counts[sol.gen_status] = counts.get(sol.gen_status, 0) + 1
            records_f.write(json.dumps(rec) + "\n"); records_f.flush()
            if sol.gen_status != "infra_error":   # truncated solutions are still scored (D12)
                samples_f.write(json.dumps({"task_id": sol.task_id, "solution": sol.text}) + "\n")
                samples_f.flush()
            done = sum(counts.values())
            tok = sol.completion_tokens if sol.completion_tokens is not None else "-"
            print(f"[{done}/{len(tasks)}] {sol.task_id}: {sol.gen_status}, {tok} tok, {sol.latency_s}s")

    if conc > 1:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            list(ex.map(one, tasks))
    else:
        for t in tasks:
            one(t)
    records_f.close()
    samples_f.close()
    print(f"Done {run_id}: {counts}")


def score(run_dir, *, ssh_host=None, local=False, skip_eval=False, dry_run=False,
          dataset="humaneval", base_only=False, image=None,
          cpus="2", memory="4g", pids_limit=256, timeout=1800, parallel=None,
          ssh_workdir="/tmp") -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bench = get_benchmark(manifest.get("benchmark", "humaneval_plus"), **manifest.get("benchmark_options", {}))
    image = image or bench.sandbox_image     # each benchmark declares its hardened scoring image
    sandbox = None
    if not skip_eval:
        sandbox = SandboxRunner(image, cpus=cpus, memory=memory, pids_limit=pids_limit,
                                read_only=bench.sandbox_read_only, auto_confirm=bench.sandbox_auto_confirm,
                                ssh_host=ssh_host, ssh_workdir=ssh_workdir, local=local, dry_run=dry_run)
    metrics = bench.evaluate(run_dir, sandbox=sandbox, dataset=dataset, base_only=base_only,
                             timeout=timeout, parallel=parallel, skip_eval=skip_eval)
    return metrics.summary


def main() -> None:
    ap = argparse.ArgumentParser(description="open-code-bench runner (generate | score)")
    sub = ap.add_subparsers(dest="op", required=True)

    g = sub.add_parser("generate", help="generate solutions for a run spec")
    g.add_argument("spec", type=Path)
    g.add_argument("--limit", type=int, default=None, help="override spec limit (0 = all)")
    g.add_argument("--concurrency", type=int, default=None, help="override spec concurrency")

    s = sub.add_parser("score", help="score a generation run dir in the sandbox")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--dataset", default="humaneval")
    s.add_argument("--image", default=None, help="override the benchmark's default sandbox image")
    s.add_argument("--ssh-host")
    s.add_argument("--ssh-workdir", default="/tmp")
    s.add_argument("--local", action="store_true")
    s.add_argument("--cpus", default="2")
    s.add_argument("--memory", default="4g")
    s.add_argument("--pids-limit", type=int, default=256)
    s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("--parallel", type=int, default=None)
    s.add_argument("--base_only", action="store_true")
    s.add_argument("--skip-eval", action="store_true")
    s.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    if args.op == "generate":
        import yaml
        spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
        generate(spec, limit=args.limit, concurrency=args.concurrency)
    else:  # score
        if not args.local and not args.ssh_host and not args.skip_eval:
            ap.error("score: pass --ssh-host <host> or --local (or --skip-eval to re-merge only)")
        summary = score(args.run_dir, ssh_host=args.ssh_host, local=args.local,
                        skip_eval=args.skip_eval, dry_run=args.dry_run, dataset=args.dataset,
                        base_only=args.base_only, image=args.image, cpus=args.cpus, memory=args.memory,
                        pids_limit=args.pids_limit, timeout=args.timeout, parallel=args.parallel,
                        ssh_workdir=args.ssh_workdir)
        if not args.dry_run:
            print("\n=== score summary ===")
            for k, v in summary.items():
                print(f"  {k}: {v:.3f}" if k == "pass@1" and isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
