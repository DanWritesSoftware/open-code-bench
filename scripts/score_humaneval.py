"""open-code-bench Phase 0 — HumanEval+ scoring (pipeline step #3).

Turns a generation run dir (runs/<run_id>/, produced by gen_humaneval.py) into per-sample
pass/fail + pass@1, in three stages:

  1. SANITIZE (local, pure text) — evalplus.sanitize strips ```python fences and trailing
     prints from each raw completion -> samples-sanitized.jsonl. No code is executed, so
     this is safe to run on the gateway host.

  2. EVALUATE (sandboxed) — evalplus.evaluate runs the sanitized solutions against the
     HumanEval+ base+plus tests INSIDE the hardened docker/exec container (D11). Untrusted,
     model-generated code never runs on the gateway host (D6, §9): by default it runs on the
     T14 sandbox host over SSH; --local uses Docker on this host as a fallback.

  3. MERGE + SCORE — join EvalPlus's <samples>_eval_results.json back onto records.jsonl by
     task_id, assign each sample an eval_status/passed per D12, compute pass@1 over the
     fairly-attempted samples plus a completeness figure, and write scores.jsonl +
     score_summary.json into the run dir.

The D11 hardening flags live at the docker-run call site (build_docker_argv), not in an
image ENTRYPOINT, so the isolation contract is auditable where it is applied.

Usage:
  # remote sandbox over SSH (primary path):
  python scripts/score_humaneval.py runs/heplus_XXXX --ssh-host <user>@<host>
  # local Docker fallback:
  python scripts/score_humaneval.py runs/heplus_XXXX --local
  # print the exact sandbox commands without running them (needs no Docker/SSH):
  python scripts/score_humaneval.py runs/heplus_XXXX --ssh-host <user>@<host> --dry-run
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

SANITIZED_NAME = "samples-sanitized.jsonl"
RESULT_NAME = "samples-sanitized_eval_results.json"   # evalplus: <samples>.jsonl -> <samples>_eval_results.json
DEFAULT_IMAGE = "ocb-exec:0.3.1"

# EvalPlus per-status constants (evalplus.eval): PASS/FAIL/TIMEOUT + None (unknown).
PASS = "pass"
TIMEOUT = "timeout"


def eval_status_for(base_status, plus_status, base_only: bool):
    """Map EvalPlus's (base_status, plus_status) onto a D12 (eval_status, passed) pair.

    EvalPlus statuses are "pass" | "fail" | "timeout" | None. NOTE: EvalPlus folds
    compile-errors and raised exceptions into "fail" -- it does not surface D12's
    failed-vs-error distinction, so a crashing solution reads here as "failed". Splitting
    those out would require parsing --test_details; deferred.
    """
    statuses = [base_status] if base_only else [base_status, plus_status]
    if any(s == TIMEOUT for s in statuses):
        return "timeout", False
    if all(s == PASS for s in statuses):
        return "passed", True
    return "failed", False


def sanitize(run_dir: Path, py: str) -> Path:
    """Strip code fences/test prints from samples.jsonl -> samples-sanitized.jsonl (local)."""
    samples = run_dir / "samples.jsonl"
    if not samples.is_file():
        sys.exit(f"error: {samples} not found (run gen_humaneval.py first)")
    print(f"[sanitize] {py} -m evalplus.sanitize {samples}")
    subprocess.run([py, "-m", "evalplus.sanitize", str(samples)], check=True)
    out = run_dir / SANITIZED_NAME
    if not out.is_file():
        sys.exit(f"error: expected {out} after sanitize, not found")
    return out


def build_docker_argv(image: str, work_mount: str, args, dataset: str) -> list[str]:
    """The hardened `docker run` invocation (D11). `work_mount` is the host dir bind-mounted
    to /work; it holds samples-sanitized.jsonl in and <...>_eval_results.json out."""
    inner = ["timeout", f"{args.timeout}s",
             "python", "-m", "evalplus.evaluate", dataset,
             "--samples", f"/work/{SANITIZED_NAME}"]
    if args.parallel:
        inner += ["--parallel", str(args.parallel)]
    if args.base_only:
        inner.append("--base_only")
    return [
        "docker", "run", "--rm",
        "--network=none",                 # untrusted code gets no network
        "--cap-drop=ALL",                 # drop every Linux capability
        "--security-opt=no-new-privileges",
        "--read-only",                    # immutable rootfs; dataset + ground-truth are pre-baked
        "--tmpfs", "/tmp:rw,size=512m",   # writable scratch for evalplus temp dirs
        "--pids-limit", str(args.pids_limit),
        "--cpus", str(args.cpus),
        "--memory", args.memory,
        "-v", f"{work_mount}:/work",
        image,
        *inner,
    ]


def run_steps(steps: list[list[str]], dry_run: bool) -> None:
    for argv in steps:
        printable = " ".join(shlex.quote(a) for a in argv)
        print(f"  $ {printable}")
        if not dry_run:
            subprocess.run(argv, check=True)


def evaluate_local(run_dir: Path, args, dataset: str) -> None:
    argv = build_docker_argv(args.image, str(run_dir.resolve()), args, dataset)
    print("[evaluate] local Docker:")
    run_steps([argv], args.dry_run)


def evaluate_ssh(run_dir: Path, args, dataset: str) -> None:
    host = args.ssh_host
    remote = f"{args.ssh_workdir.rstrip('/')}/ocb-score-{run_dir.name}"
    local_samples = run_dir / SANITIZED_NAME
    local_result = run_dir / RESULT_NAME
    docker_argv = build_docker_argv(args.image, remote, args, dataset)
    docker_cmd = " ".join(shlex.quote(a) for a in docker_argv)
    print(f"[evaluate] T14 sandbox over SSH: {host}  (remote work dir: {remote})")
    steps = [
        ["ssh", host, f"mkdir -p {shlex.quote(remote)} && chmod 777 {shlex.quote(remote)}"],
        ["scp", str(local_samples), f"{host}:{remote}/{SANITIZED_NAME}"],
        ["ssh", host, docker_cmd],
        ["scp", f"{host}:{remote}/{RESULT_NAME}", str(local_result)],
        ["ssh", host, f"rm -rf {shlex.quote(remote)}"],
    ]
    run_steps(steps, args.dry_run)


def merge_and_score(run_dir: Path, dataset: str, base_only: bool) -> dict:
    """Join EvalPlus results onto records.jsonl, assign D12 eval_status, compute pass@1."""
    result_path = run_dir / RESULT_NAME
    results = json.loads(result_path.read_text(encoding="utf-8"))
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

        # EvalPlus stores per-task results sorted by completion_id; sample_index maps 1:1
        # to that order (the runner writes samples in (task, index) order).
        task_entries = eval_by_task.get(tid, [])
        if idx >= len(task_entries):
            rows.append({**rec, "eval_status": "sandbox_error", "passed": None})
            continue
        entry = task_entries[idx]
        eval_status, passed = eval_status_for(entry.get("base_status"),
                                              entry.get("plus_status"), base_only)
        n_scored += 1
        if gen_status == "ok":           # pass@1 is over fairly-attempted samples (D12)
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
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a HumanEval+ generation run (step #3).")
    ap.add_argument("run_dir", type=Path, help="runs/<run_id>/ from gen_humaneval.py")
    ap.add_argument("--dataset", default="humaneval", help="evalplus dataset (humaneval)")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="hardened sandbox image tag")
    # sandbox target: SSH (primary) vs local Docker fallback
    ap.add_argument("--ssh-host", help="user@host of the T14 sandbox (primary path)")
    ap.add_argument("--ssh-workdir", default="/tmp", help="base dir on the T14 for scratch")
    ap.add_argument("--local", action="store_true", help="run Docker on this host instead of SSH")
    # D11 resource caps
    ap.add_argument("--cpus", default="2", help="container CPU cap")
    ap.add_argument("--memory", default="4g", help="container memory cap")
    ap.add_argument("--pids-limit", type=int, default=256, help="container PID cap")
    ap.add_argument("--timeout", type=int, default=1800, help="whole-run wall clock (s)")
    ap.add_argument("--parallel", type=int, default=None, help="evalplus workers (default: cpu//2)")
    ap.add_argument("--base_only", action="store_true", help="score base tests only (skip the + tests)")
    ap.add_argument("--skip-eval", action="store_true", help="merge an existing eval_results.json only")
    ap.add_argument("--dry-run", action="store_true", help="print sandbox commands, do not execute")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"error: {run_dir} is not a directory")
    if not args.local and not args.ssh_host and not args.skip_eval:
        sys.exit("error: choose a sandbox target: --ssh-host <user@t14> (primary) or --local")

    py = sys.executable
    if not args.skip_eval:
        sanitize(run_dir, py)
        if args.ssh_host:
            evaluate_ssh(run_dir, args, args.dataset)
        else:
            evaluate_local(run_dir, args, args.dataset)

    if args.dry_run:
        print("\n[dry-run] sandbox not executed; skipping merge/score.")
        return

    summary = merge_and_score(run_dir, args.dataset, args.base_only)
    print("\n=== score summary ===")
    for k, v in summary.items():
        if k == "pass@1":
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWrote {run_dir / 'scores.jsonl'} and {run_dir / 'score_summary.json'}")


if __name__ == "__main__":
    main()
