# open-code-bench

Run multiple coding benchmarks against multiple LLM backends through one gateway.

A [LiteLLM](https://docs.litellm.ai/) proxy presents a single OpenAI-compatible API. Untrusted, model-generated code is executed only inside a hardened, network-isolated Docker sandbox.


## Results

Two benchmarks across four self-hosted Qwen backends. All runs: temperature 0 (greedy), single sample (pass@1). 
*Completeness* = fairly-attempted / total
(truncated or infra-errored samples are excluded from pass@1, never scored as wrong).

### HumanEval+ — 164 tasks

| Model | Backend | pass@1 | base pass@1 | Completeness |
|---|---|---|---|---|
| qwen2.5-coder:1.5b (Q4_K_M) | Raspberry Pi 5 · Ollama | 0.610 | 0.665 | 164/164 |
| qwen2.5-coder:7b | DGX Spark · vLLM (GB10) | 0.823 | 0.872 | 164/164 |
| **qwen2.5-coder:32b** | DGX Spark · vLLM (GB10) | **0.866** | **0.909** | 164/164 |
| qwen2.5-72b-instruct (AWQ 4-bit) | DGX Spark · vLLM (GB10) | 0.805 | 0.848 | 164/164 |

<sub>pass@1 = HumanEval+ (base + extra tests); *base pass@1* = original HumanEval tests only.
Dataset: HumanEvalPlus v0.1.10 (`fe585eb4…`), EvalPlus 0.3.1. Runs `heplus_20260623T163736Z` (1.5B),
`heplus_20260623T214900Z` (7B), `heplus_20260623T230559Z` (32B), `heplus_20260624T222256Z` (72B).</sub>

### BigCodeBench-hard — 148 tasks (complete split)

| Model | Backend | pass@1 | Completeness |
|---|---|---|---|
| qwen2.5-coder:7b | DGX Spark · vLLM (GB10) | 0.224 | 147/148 |
| **qwen2.5-coder:32b** | DGX Spark · vLLM (GB10) | **0.385** | 148/148 |
| qwen2.5-72b-instruct (AWQ 4-bit) | DGX Spark · vLLM (GB10) | 0.324 | 148/148 |

<sub>pass@1 over fairly-attempted (matches the official BigCodeBench scorer within rounding — e.g.
7B = 0.224 here vs 0.223 over all 148). *complete* split = fill in the function body from its
signature + docstring. Dataset: BigCodeBench v0.1.4 hard subset (`f8d6f960…`), bigcodebench 0.2.5.
Runs `bcb_20260626T170715Z` (7B), `bcb_20260626T211347Z` (32B), `bcb_20260626T215636Z` (72B).</sub>

### Key findings

- **BigCodeBench-hard discriminates far better than HumanEval+.** The 7B→32B jump is **+72%** on
  BCB-hard (0.224→0.385) versus just **+5%** on HumanEval+ (0.823→0.866). HumanEval+ is near-saturated
  for capable coders — BCB-hard has real headroom and is the more useful ranking benchmark here.
- **Same ranking on both benchmarks: 32B-Coder > 72B-Instruct-AWQ > 7B-Coder.** The 72B is a *general*
  Instruct model at 4-bit AWQ (not code-specialized) and lands below the 32B-**Coder** on both —
  code-specialization + full precision beats raw size + quantization for coding.
- **Edge device:** the 1.5B on a Raspberry Pi 5 reaches 0.610 on HumanEval+ — respectable for a CPU
  edge box. 


## How it works

```
run → gateway (generate, tagged by run_id) → sandbox (run tests, score) → results + provenance
```

1. **Generate** — the runner calls the gateway (OpenAI protocol), which routes to the chosen
   backend. Every call is tagged with the `run_id`, model, sampling params, and dataset version.
2. **Score** — generated solutions are sanitized, then run against the benchmark's tests inside
   the hardened sandbox container (`--network=none`, dropped capabilities, resource caps; read-only
   rootfs where the evaluator allows it). Scoring is delegated to the benchmark's native evaluator
   (EvalPlus for HumanEval+, BigCodeBench for BCB-hard).
3. **Results** — per-sample pass/fail, `pass@1`, and a completeness figure, written alongside
   full provenance (model revision, sampling, dataset hash, tool versions, git commit).

## Quickstart

Prerequisites: Python 3.13 + a venv (see [Setup](#setup)); the backends in `litellm/config.yaml`
reachable from the gateway; and a sandbox host over SSH with Docker and the scoring image for your
benchmark built (`docker/exec/` for HumanEval+, `docker/exec-bcb/` for BigCodeBench).

A run is two spec-driven steps. **generate** talks to the gateway/backends; **score** runs in the
sandbox. They're decoupled, so they can run on different hosts or networks — generate where the model
backends live, then score where the sandbox lives.

```powershell
# 1. Start the gateway (Windows; routes each logical model to its backend)
.\scripts\serve-gateway.ps1                          # serves http://127.0.0.1:4000
$env:PYTHONUTF8 = '1'

# 2. Generate — model, sampling, and concurrency are all declared in the spec
.\.venv\Scripts\python.exe -m ocb.runner.run generate specs\bigcodebench-hard.yaml
#   -> runs\bcb_<timestamp>\   (manifest.json, records.jsonl, samples.jsonl)
#   specs\heplus.yaml runs HumanEval+ across three models -> one runs\heplus_<timestamp>\ per model

# 3. Score a run in the sandbox (the benchmark's hardened image is selected automatically)
.\.venv\Scripts\python.exe -m ocb.runner.run score runs\bcb_<timestamp> --ssh-host sandbox
#   -> scores.jsonl + score_summary.json with the real pass@1
```

`generate` flags: `--limit N` (first N tasks — but HumanEval+ scoring needs all 164, so leave it 0),
`--concurrency C` (override the spec; keep 1 for the Pi). `score` flags: `--local` (Docker on this
host instead of SSH), `--dry-run` (print the exact hardened `docker run` without executing),
`--skip-eval` (re-merge existing eval results). The `scripts/gen_humaneval.py` and
`scripts/score_humaneval.py` entry points still work — they're thin wrappers over the runner.

## Layout

```
litellm/config.yaml          gateway model_list (the backends)
specs/                       run specs (benchmark + models + sampling); one per benchmark/model-set
src/ocb/
  gateway/client.py          OpenAI-protocol client; tags every call with run_id + metadata
  benchmarks/                Benchmark plugins: base.py (ABC) + humaneval_plus.py + bigcodebench.py
  sandbox/runner.py          hardened `docker run` (local or over SSH), scp samples in / results out
  runner/run.py              spec-driven entrypoint: generate | score
  provenance.py              dataset hash, git commit, tool versions
docker/exec/                 HumanEval+ sandbox image (EvalPlus + dataset pre-baked, offline)
docker/exec-bcb/             BigCodeBench sandbox image (139 task libs + dataset pre-baked, offline)
scripts/serve-gateway.ps1    start the gateway
scripts/gen_humaneval.py     legacy wrapper over `runner generate` (HumanEval+)
scripts/score_humaneval.py   legacy wrapper over `runner score`
runs/                        per-run artifacts + provenance (gitignored)
```

Adding a benchmark = one plugin under `src/ocb/benchmarks/` (subclass `Benchmark`: build prompts,
extract solutions, delegate scoring to its native evaluator) + a spec. See `bigcodebench.py` for a
worked second example.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
# .venv/bin/python  -m pip install -r requirements.txt    # Linux/macOS
.venv\Scripts\python -m pip install -e . --no-deps         # editable install of the `ocb` package
```

On Windows, set `PYTHONUTF8=1` before running the gateway or the runner (LiteLLM's startup banner
crashes under the cp1252 console codepage otherwise). On the sandbox host, build the image(s) for the
benchmarks you'll score, once:

```bash
docker build -t ocb-exec:0.3.1 docker/exec            # HumanEval+
docker build -t ocb-bcb-exec:0.2.4 docker/exec-bcb    # BigCodeBench
```
