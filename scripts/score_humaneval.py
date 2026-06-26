"""Thin wrapper — HumanEval+ scoring now lives in the framework (ocb.runner + the plugin).

Kept for the familiar CLI; delegates to the runner, which runs the benchmark's evaluator inside
the hardened sandbox (D11). New code should prefer:

    python -m ocb.runner.run score runs/<run_id> --ssh-host sandbox

Flags: --ssh-host <host> (primary) | --local (Docker here) | --skip-eval (re-merge existing
results) | --dry-run (print the hardened docker run without executing). Windows: PYTHONUTF8=1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocb.runner.run import score


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a HumanEval+ run in the sandbox (wrapper over ocb.runner).")
    ap.add_argument("run_dir", type=Path, help="runs/<run_id>/ from generation")
    ap.add_argument("--dataset", default="humaneval")
    ap.add_argument("--image", default="ocb-exec:0.3.1")
    ap.add_argument("--ssh-host", help="user@host of the sandbox (primary path)")
    ap.add_argument("--ssh-workdir", default="/tmp")
    ap.add_argument("--local", action="store_true", help="run Docker on this host instead of SSH")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--pids-limit", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--skip-eval", action="store_true", help="re-merge an existing eval_results only")
    ap.add_argument("--dry-run", action="store_true", help="print the sandbox commands, do not execute")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        sys.exit(f"error: {args.run_dir} is not a directory")
    if not args.local and not args.ssh_host and not args.skip_eval:
        sys.exit("error: choose a sandbox target: --ssh-host <host> (primary) or --local")

    summary = score(args.run_dir, ssh_host=args.ssh_host, local=args.local, skip_eval=args.skip_eval,
                    dry_run=args.dry_run, dataset=args.dataset, base_only=args.base_only,
                    image=args.image, cpus=args.cpus, memory=args.memory, pids_limit=args.pids_limit,
                    timeout=args.timeout, parallel=args.parallel, ssh_workdir=args.ssh_workdir)
    if not args.dry_run:
        print("\n=== score summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v:.3f}" if k == "pass@1" and isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
