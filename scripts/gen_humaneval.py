"""Thin wrapper — HumanEval+ generation now lives in the framework (ocb.runner + the plugin).

Kept for the familiar CLI; builds a one-model run spec and calls the runner. New code should
prefer the spec-driven entrypoint:

    python -m ocb.runner.run generate specs/heplus.yaml

Run (Windows):  $env:PYTHONUTF8='1'; .venv\\Scripts\\python.exe scripts\\gen_humaneval.py --model qwen-spark-7b --limit 0 --concurrency 32
"""
from __future__ import annotations

import argparse

from ocb.runner.run import GATEWAY, generate


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate HumanEval+ solutions via the gateway (wrapper over ocb.runner).")
    ap.add_argument("--model", default="qwen-pi-1.5b", help="gateway logical model name")
    ap.add_argument("--limit", type=int, default=5, help="number of problems (0 = all 164)")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context window")
    ap.add_argument("--concurrency", type=int, default=1, help="concurrent in-flight requests (>1 lets vLLM batch)")
    args = ap.parse_args()

    spec = {
        "benchmark": "humaneval_plus",
        "models": [args.model],
        "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens, "num_ctx": args.num_ctx},
        "limit": args.limit,
        "concurrency": args.concurrency,
        "gateway": GATEWAY,
    }
    generate(spec)


if __name__ == "__main__":
    main()
