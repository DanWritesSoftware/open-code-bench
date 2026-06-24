# open-code-bench

Run multiple coding benchmarks against multiple LLM backends through one gateway.

A [LiteLLM](https://docs.litellm.ai/) proxy presents a single OpenAI-compatible API. Untrusted, model-generated code is executed only inside a hardened, network-isolated Docker sandbox.


## Results

| Model | Backend | Benchmark | pass@1 | base pass@1 | Completeness |
|---|---|---|---|---|---|
| qwen2.5-coder:1.5b (Q4_K_M) | Raspberry Pi 5 · Ollama | HumanEval+ | **0.610** | 0.665 | 164/164 |
| qwen2.5-coder:7b | DGX Spark · vLLM (GB10) | HumanEval+ | **0.823** | 0.872 | 164/164 |
| qwen2.5-coder:32b | DGX Spark · vLLM (GB10) | HumanEval+ | **0.866** | 0.909 | 164/164 |

<sub>pass@1 = HumanEval+ (base + extra tests); *base pass@1* = original HumanEval tests only. temp=0
(greedy), single sample. Self-hosted backends — wall cost not metered. Dataset: HumanEvalPlus
v0.1.10 (`fe585eb4…`), EvalPlus 0.3.1. Runs: `heplus_20260623T163736Z` (Pi 1.5B),
`heplus_20260623T214900Z` (Spark 7B), `heplus_20260623T230559Z` (Spark 32B). Completeness =
fairly-attempted / total (no infra/truncation).</sub>


## How it works

```
run → gateway (generate, tagged by run_id) → sandbox (run tests, score) → results + provenance
```

1. **Generate** — the runner calls the gateway (OpenAI protocol), which routes to the chosen
   backend. Every call is tagged with the `run_id`, model, sampling params, and dataset version.
2. **Score** — generated solutions are sanitized, then run against the benchmark's tests inside
   the hardened sandbox container (`--network=none`, read-only rootfs, dropped capabilities,
   resource caps). Scoring is delegated to the benchmark's native evaluator (EvalPlus).
3. **Results** — per-sample pass/fail, `pass@1`, and a completeness figure, written alongside
   full provenance (model revision, sampling, dataset hash, tool versions, git commit).

## Quickstart

Prerequisites: Python 3.13, a venv with deps installed (see [Setup](#setup)), and a sandbox host
reachable over SSH with Docker + the `ocb-exec` image built (see `docker/exec/`).

```powershell
# 1. Start the gateway (Windows; routes qwen-pi-1.5b -> Pi Ollama)
.\scripts\serve-gateway.ps1                      # serves http://127.0.0.1:4000

# 2. Generate HumanEval+ solutions (all 164 tasks; --limit N for a subset)
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\gen_humaneval.py --limit 0
#   -> runs\heplus_<timestamp>\  (manifest.json, records.jsonl, samples.jsonl)

# 3. Score that run inside the sandbox (full 164-task run required by EvalPlus)
.\.venv\Scripts\python.exe scripts\score_humaneval.py runs\heplus_<timestamp> --ssh-host sandbox
#   -> scores.jsonl + score_summary.json with the real pass@1
```

`score_humaneval.py` flags: `--local` (Docker on this host instead of SSH), `--dry-run` (print the
exact hardened `docker run` without executing), `--skip-eval` (re-merge existing results).

## Status

**Phase 0 complete** — the full vertical slice works end-to-end: gateway → generate → sandbox
score → attributed results, demonstrated with HumanEval+ on the Pi backend (above).

Next: README leaderboard automation, stronger/faster backends (Spark vLLM `qwen-spark-32b`),
a Postgres results store, and more benchmarks (BigCodeBench, LiveCodeBench, Aider polyglot).

## Layout

```
litellm/config.yaml        gateway model_list (the backends)
scripts/serve-gateway.ps1  start the gateway
scripts/gen_humaneval.py   generate solutions via the gateway
scripts/score_humaneval.py sanitize -> sandbox-evaluate -> pass@1
docker/exec/               hardened sandbox image (EvalPlus + dataset pre-baked)
runs/                      per-run artifacts + provenance (gitignored)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
# .venv/bin/python  -m pip install -r requirements.txt    # Linux/macOS
.venv\Scripts\python -m pip install -e . --no-deps         # editable install of the `ocb` package
```

On Windows, set `PYTHONUTF8=1` before running the gateway or scripts (LiteLLM's startup banner
crashes under the cp1252 console codepage otherwise). On the sandbox host, build the image once:
`docker build -t ocb-exec:0.3.1 docker/exec`.
