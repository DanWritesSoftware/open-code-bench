# open-code-bench — Architecture

**Status:** Draft / proposed
**Last updated:** 2026-06-16

---

## 1. Purpose & scope

open-code-bench is an evaluation platform that runs **multiple coding benchmarks**
against **multiple, heterogeneous LLM endpoints** through a **single gateway**.

The gateway is a [LiteLLM](https://docs.litellm.ai/) proxy that presents one
OpenAI-compatible API. Benchmarks speak that one protocol and never need to know
whether a model is a small Qwen-Coder running on a Raspberry Pi, a larger
Qwen-Coder on an NVIDIA DGX Spark, a model on Amazon Bedrock, or the Anthropic
Claude API. Adding a model becomes a config entry; adding a benchmark becomes a
plugin that targets the gateway.

The guiding principle: **decouple the two axes** — *what model is being
evaluated* (behind the gateway) and *which benchmark is doing the evaluating* (in
front of it) — so each can grow independently.

## 2. Goals & non-goals

**Goals**

- One OpenAI-compatible front door for all models (self-hosted + cloud).
- Pluggable benchmark layer: HumanEval+ first, then LiveCodeBench, BigCodeBench,
  Aider polyglot, with the same machinery.
- Reproducible, fully-attributed runs: every result is traceable to a model
  revision, sampling params, benchmark version, and dataset hash.
- Cost / token / latency tracking joined to benchmark accuracy, per run.
- Safe execution of untrusted, model-generated code.

**Non-goals (for now)**

- A hosted multi-tenant service. This is an internal evaluation tool.
- Training / fine-tuning. We evaluate models, we don't produce them.
- A custom web UI in the first iterations (reporting starts as tables/artifacts).

## 3. System overview

```
                       run spec (YAML): benchmark + models + sampling
                                     │
                          ┌──────────▼───────────┐
                          │   Orchestrator/Runner │  assigns run_id, fans out
                          └───┬───────────────┬───┘
              ensure-up (SSH) │               │ generate (OpenAI proto, tagged run_id)
                 ┌────────────▼──┐     ┌───────▼──────────────────┐
                 │ Endpoint mgr  │     │   LiteLLM Proxy (gateway) │ routing / keys /
                 │ Pi: ollama up │     └──┬────────┬────────┬──────┘ retries / cost / log
                 │ Spark: vllm up│     HTTP│ tunnel │ Bedrock│ Anthropic
                 └───────────────┘   ┌─────▼──┐ ┌───▼────┐ ┌─▼─────────┐
                                     │Pi Ollama│ │Spark   │ │cloud APIs │
                                     │ Qwen sm │ │vLLM Qwen│ │           │
                                     └─────────┘ └────────┘ └───────────┘
                                     │
   completions ──> ┌────────────────▼─────────────┐
                   │  Benchmark adapter (per bench) │ prompt build + answer extract
                   └────────────────┬─────────────┘
                                    │ candidate solutions
                   ┌────────────────▼─────────────┐
                   │  Sandbox / evaluator (Docker) │ untrusted code exec → pass/fail
                   └────────────────┬─────────────┘
                                    │ per-task results + metrics
                   ┌────────────────▼─────────────┐
                   │  Results store + reporting    │ join w/ LiteLLM cost/latency by run_id
                   └──────────────────────────────┘
```

Two control paths to the self-hosted boxes:

- **Data plane:** HTTP to an OpenAI-compatible inference server on each box,
  over the LAN (or via an SSH local-forward tunnel if inference ports must stay
  closed). The gateway only ever speaks HTTP to a backend.
- **Control plane:** SSH, used *only* to start/stop the inference server and
  load/pull the right model before a run. SSH is never on the inference data
  path.

## 4. Endpoints (model backends)

All four are registered as logical model names in the LiteLLM `model_list`.

| Logical name (example) | Backend | Server | Notes |
|---|---|---|---|
| `qwen-pi-1.5b` | Raspberry Pi 5 (8 GB, CPU) | Ollama / llama.cpp | Qwen2.5-Coder 0.5B/1.5B (good), 3B (slow). An **edge data point**, not a throughput target. Concurrency 1–2. |
| `qwen-spark-32b` | NVIDIA DGX Spark (~128 GB unified, ARM64) | **vLLM** | Qwen2.5-Coder 7B/14B/32B (and Qwen3-Coder-30B-A3B). Primary self-hosted model; vLLM batching for parallel single-shot calls. |
| `bedrock-<model>` | Amazon Bedrock | LiteLLM `bedrock/` | AWS creds / IAM on the gateway host only. |
| `claude-<model>` | Anthropic Claude API | LiteLLM `anthropic/` | API key on the gateway host only. |

Illustrative LiteLLM config fragment (not final):

```yaml
model_list:
  - model_name: qwen-spark-32b
    litellm_params:
      model: hosted_vllm/Qwen2.5-Coder-32B-Instruct
      api_base: http://spark.lan:8000/v1
      rpm: 600            # tune to vLLM capacity
  - model_name: qwen-pi-1.5b
    litellm_params:
      model: ollama_chat/qwen2.5-coder:1.5b   # ollama_chat -> /api/chat, applies the model's chat template
      api_base: http://pi.lan:11434
      rpm: 30             # Pi is slow; keep concurrency low
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
  - model_name: bedrock-nova-pro
    litellm_params:
      model: bedrock/amazon.nova-pro-v1:0
```

## 5. Components

### 5.1 LiteLLM Proxy (the gateway)
Config-driven `model_list` mapping logical names to backends. Owns:
- routing + **bounded transport-only retries (D9)** + timeouts + per-model
  rate/concurrency limits; response **caching and cross-model fallbacks are
  disabled for eval runs**,
- **all provider secrets** (AWS, Anthropic) — benchmarks receive only a virtual
  key, never raw cloud credentials,
- structured logging of every call (request, model, tokens, cost, latency) to
  Postgres, keyed by request metadata.

### 5.2 Endpoint manager (SSH, control plane)
"Ensure model *X* is up and healthy on box *Y*" before a run. SSHes into the Pi
or Spark, starts the inference server, pulls/loads the requested model, then
waits for **readiness** — a successful *test generation*, not a one-shot `/v1`
ping, because "process up" ≠ "model resident and serving" (a 32B vLLM load takes
minutes). Model swaps on single-model-per-server boxes are **serialized per box**
(D13). No-op for cloud backends. This is the **only** SSH in the system and it
never carries inference traffic.

### 5.3 Benchmark adapters (the plugin layer)
Each benchmark implements a small interface (see §6). The adapter is responsible
for *building prompts* and *extracting solutions*; it delegates *scoring* to the
benchmark's native evaluator where one exists.

### 5.4 Sandbox / evaluator (Docker)
Isolated execution of untrusted, model-generated code. **Required from day one**
(HumanEval+ runs generated code). Standardized so every future benchmark reuses
the same isolation boundary (D11).

**Our container is the boundary.** Each evaluation runs in a hardened container:
non-root, `--network=none`, read-only root filesystem + a writable `tmpfs`
scratch, `--cap-drop=ALL`, `--pids-limit`, a per-test wall-clock timeout, and
`--cpus` / `--memory` caps (an OOM is a task `error`, not a harness crash).
Benchmark scorers (e.g. EvalPlus) run *inside* this container — we delegate
scoring (D1) but never treat a tool's built-in execution as the security
boundary, and we never nest a second Docker layer.

**Two concurrency knobs, decoupled:** generation concurrency is gated by the
model backend; evaluation concurrency is gated by the sandbox host's cores/RAM.

**Two profiles:** `exec` (single-file: HumanEval+, BigCodeBench) and `workspace`
(a per-task writable repo checkout + multiple language toolchains + multi-turn
edits, for agentic benchmarks such as Aider polyglot). Phase 0 needs only
`exec`; the abstraction is shaped for both now. The Phase-0 sandbox host is the
x86 workstation (Docker Desktop), so EvalPlus's prebuilt image is usable as-is.

### 5.5 Results store + reporting
**Postgres is the system of record (D7)** — the same instance LiteLLM already
requires, but in our own schema, never inside LiteLLM's internal tables. Cost,
tokens, and latency are copied from each response into the per-sample row at
write time, so results are self-contained and immune to LiteLLM version drift;
LiteLLM's spend log is a read-only cross-check. JSONL artifacts + a run manifest
are still written to `runs/` as a portable export.

```sql
run     (run_id PK, spec jsonb, benchmark, benchmark_version, tool_version,
         dataset_version, dataset_hash, model_logical, sampling jsonb,
         gateway_version, git_commit, started_at, finished_at, status)

sample  (run_id, task_id, sample_index)  PK,
         litellm_call_id, prompt_hash, raw_completion, extracted_solution,
         prompt_tokens, completion_tokens, cost_usd, latency_ms,
         gen_status, eval_status, passed, eval_detail jsonb, created_at
```

`pass@k` and the `model × benchmark × accuracy × $/task × latency` tables are SQL
views aggregating `sample`.

### 5.6 Orchestrator / Runner
Single entrypoint. Reads a declarative run spec, calls the endpoint manager to
bring backends up, drives the benchmark adapter against the gateway, scores via
the sandbox, and writes results. Assigns the `run_id` that ties everything
together.

## 6. Extension model (the benchmark seams)

The roadmap benchmarks were chosen because each adds exactly one new dimension.
Designing for these four now means later additions are "implement the interface,"
not "re-architect."

| Benchmark | New dimension | Seam exercised |
|---|---|---|
| **HumanEval+** | baseline: 1 prompt → 1 completion → run unit tests | generation client + sandbox runner |
| **BigCodeBench** | richer prompts, library imports, heavier exec env | same seams, more sandbox deps |
| **LiveCodeBench** | stdin/stdout judging; time-windowed dataset versions | pluggable evaluator + dataset versioning |
| **Aider polyglot** | multi-turn / agentic, file edits, multi-language, repo state | multi-turn loop + workspace/repo sandbox |

The four extension points that fall out of this:
1. **Generation client** — shared; talks to the gateway, tags every call.
2. **Prompt/format** — per benchmark.
3. **Evaluator** — per benchmark (delegated to native tooling where possible).
4. **Conversation mode** — single-shot vs multi-turn/agentic, expressed through
   the `run()` driver (D14).

Sketch of the `Benchmark` interface (illustrative):

```python
class Benchmark(ABC):
    name: str
    version: str
    conversation_mode: Literal["single_shot", "multi_turn"]

    @abstractmethod
    def load_dataset(self) -> list[Task]: ...

    # single-shot building blocks (used by the default run())
    def build_prompt(self, task: Task) -> list[Message]: ...
    def extract_solution(self, completion: str, task: Task) -> Solution: ...

    # driver: owns the model interaction; default == single-shot (D14)
    def run(self, task: Task, client: GatewayClient) -> Solution:
        completion = client.complete(self.build_prompt(task), **self.sampling)
        return self.extract_solution(completion, task)

    @abstractmethod
    def evaluate(self, task: Task, solution: Solution) -> TaskResult: ...

    def aggregate(self, results: list[TaskResult]) -> Metrics:  # e.g. pass@k
        ...
```

HumanEval+ implements `load_dataset`/`build_prompt`/`extract_solution`/`evaluate`
and inherits the default `run()`. Agentic benchmarks (Aider) override `run()`.

## 7. Key design decisions

**D1 — Own generation, delegate scoring.**
Generation always flows through *our* gateway client so cost, latency,
provenance, concurrency, and `run_id` tagging are uniform across every
benchmark. Scoring is delegated to each benchmark's trusted native evaluator
(e.g. EvalPlus for HumanEval+) rather than reimplemented. Split: *we own
generation, the benchmark owns scoring.*

**D2 — SSH is control plane only.**
Inference is HTTP over the LAN to an OpenAI-compatible server on each box. SSH
starts servers and loads models. This keeps benchmarks oblivious to transport
and avoids a brittle SSH-stdio inference path.

**D3 — vLLM on Spark, Ollama on Pi.**
HumanEval+ issues many parallel single-shot calls; vLLM's batching matters on
the Spark. The Pi is a slow edge data point regardless, so Ollama's simplicity
(on-demand model loading, easy multi-model) wins there.

**D4 — Centralized secrets.**
Cloud credentials live only on the gateway host. Benchmark processes hold only a
LiteLLM virtual key. Rotating a cloud key touches one place.

**D5 — Provenance by default.**
Every run records model id + revision, sampling params, benchmark name +
version, dataset hash, gateway version, git commit, and timestamp. Reproducible
scoring is cheap if it's built in from the start and expensive to retrofit.

**D6 — Sandbox from day one.**
Even HumanEval+ executes untrusted code; isolation is not deferred to "later
benchmarks."

**D7 — One store: Postgres, our schema, denormalized cost.**
Results live in the same Postgres instance LiteLLM already requires, but in our
own tables — never inside LiteLLM's internal schema. Cost, tokens, and latency
are copied from each response into the per-sample row at write time, so results
are self-contained and immune to LiteLLM version drift; LiteLLM's spend log is a
read-only cross-check. SQLite is dropped.

**D8 — Sampling is runner-driven.**
k samples are k independent single-shot calls issued by the runner, not the
provider `n` parameter (which Anthropic lacks and Ollama implements unreliably).
Each sample is separately tagged and logged, so cost is exactly k×.
`temperature/top_p/max_tokens/base_seed` are recorded per run; per-sample
`seed = base_seed + index` where the backend supports it. Seeds and temp=0 are
recorded for provenance, not promised as bitwise-reproducible across
heterogeneous backends.

**D9 — Eval-grade request semantics.**
For eval runs: response caching is OFF; retries fire only on transport failures
(connection / timeout / 429 / 5xx with no body) and never on a returned
completion; cross-model fallbacks are disabled. A returned completion —
including empty, truncated, or refusal — is the sample and is scored as-is. A
transport failure that exhausts retries becomes an `infra_error` sample (D12),
not a wrong answer.

**D10 — Self-hosted cost is labeled, not zeroed.**
`$/task` reports measured cloud spend; self-hosted backends render as
`self-hosted` (never $0). A `tok/s` efficiency column accompanies accuracy for
all backends, and `Wh/task` is added where power metering exists. Synthetic
`est. $/task` is opt-in and always annotated with its amortization and power
assumptions. *(Applied with this default; the synthetic-$ and power-metering
choices remain open — §12.)*

**D11 — The sandbox is our isolation boundary; scoring runs inside it.**
Generated code executes only in our hardened container — non-root,
`--network=none`, read-only rootfs + tmpfs scratch, `--cap-drop=ALL`,
`--pids-limit`, per-test wall-clock + `--cpus` + `--memory` caps. Benchmark
scorers (EvalPlus) run *inside* this container (D1); we never treat a tool's
built-in execution as the security boundary, and never nest a second Docker.
Generation and evaluation concurrency are separate knobs. Two profiles —
`exec` (single-file) and `workspace` (writable repo + multi-language toolchains,
for agentic benchmarks).

**D12 — Infra failure ≠ wrong answer; runs resume on `(run_id, task_id, sample_index)`.**
Per-sample `gen_status` and `eval_status` (§8) separate model behavior from
infrastructure failure. pass@k is computed only over fairly-attempted samples;
infra failures are excluded from accuracy and reported as a completeness figure.
Runs are resumable and idempotent: the runner regenerates only missing samples
and re-attempts infra failures, but never regenerates a completed (`ok`) sample.
Every run reports both its score and its completeness.

**D13 — One active model per single-model box; swaps serialized and amortized.**
vLLM serves one model per process (swap = restart + multi-minute reload); Ollama
swaps on demand. The endpoint manager serializes swaps per box behind a lock,
and "ready" means a successful test generation, not a one-shot health ping.
Model-matrix sweeps are scheduled grouped by `(box, model)` so each costly load
happens once per model. Co-locating multiple vLLM instances on the Spark is
possible at low precision but off by default. *(Applied with this default; Spark
precision and co-location remain open — §12.)*

**D14 — A `run()` driver generalizes the interaction loop.**
The harness calls `run(task, client)`; the default implements the single-shot
path from `build_prompt`/`extract_solution` (HumanEval+ unchanged). Multi-turn /
agentic benchmarks override `run()` to drive their own edit→apply→test loop.
Scoring stays in `evaluate()` (D1). The runner invokes `run()` k times per task
(D8).

**D15 — Pin datasets and tools from Phase 0.**
Every run records the pinned evaluator/tool version, the dataset version, and a
content hash of the loaded dataset — for HumanEval+ now, not deferred to
LiveCodeBench. These fields extend D5 provenance and live in the run manifest
and the `run` table. LiveCodeBench's time-windowing is one instance of this
policy.

## 8. Request tagging & data model

Every gateway call carries:

```json
{ "metadata": { "run_id": "...", "benchmark": "humaneval_plus",
                "benchmark_version": "...", "dataset_version": "...",
                "task_id": "...", "sample_index": 0,
                "model_logical": "qwen-spark-32b" } }
```

LiteLLM persists this alongside token/cost/latency. Our `sample` table keys
per-sample outcomes by `(run_id, task_id, sample_index)` (D7); a join with
LiteLLM's spend log cross-checks cloud cost. A **run manifest** captures the full
spec + provenance — including pinned tool/dataset versions and the dataset hash
(D5, D15) — so any run can be reproduced or audited.

**Result states (D12).** Two state machines separate model behavior from
infrastructure failure:

- `gen_status`: `ok` | `infra_error` (transport / timeout / backend-down after
  the D9 retries) | `skipped`.
- `eval_status` (only when `gen_status = ok`): `passed` | `failed` (tests ran,
  wrong) | `error` (code raised / won't compile — a *model* failure) | `timeout`
  (exceeded the sandbox wall-clock — not-passed, flagged) | `sandbox_error`
  (container failed / harness OOM — *infrastructure*, not the model).

pass@k is computed **only** over fairly-attempted samples (`gen_status = ok` and
`eval_status ∈ {passed, failed, error, timeout}`). `infra_error` and
`sandbox_error` are excluded from accuracy and reported as a separate
completeness figure (e.g. "162/164 scored, 2 infra"). Runs are resumable and
idempotent on `(run_id, task_id, sample_index)`: on restart the runner
regenerates only missing samples and re-attempts infra failures, never
regenerating a completed (`ok`) sample.

## 9. Security & isolation

- The gateway and untrusted code execution sit at opposing trust levels: the
  gateway holds cloud secrets (AWS/Anthropic), so a sandbox escape on the same
  host risks credential exfiltration. **Target rule: code execution does not run
  on the gateway host.**
- **Phase 0 reality:** a single workstation runs *both* the gateway and code
  execution. The rule is enforced at the container boundary instead of by host
  separation — cloud secrets live only in the gateway container's environment;
  the sandbox runs as a separate container with no secret access, `--network=none`,
  and the D11 limits. This is the explicit, weaker Phase-0 minimum; moving code
  execution to a separate host remains the target as the system grows.
- Cloud secrets never leave the gateway container (§7 D4).
- If inference ports cannot be exposed on the LAN, backends bind to `localhost`
  and the gateway reaches them via `autossh -L` tunnels.

## 10. Repository layout

```
open-code-bench/
  docker-compose.yml          # litellm + postgres + sandbox runner
  litellm/config.yaml         # model_list (the 4 backends)
  src/ocb/
    gateway/                  # OpenAI client wrapper -> gateway, run_id tagging
    endpoints/                # SSH orchestration: pi.py, spark.py, base.py
    benchmarks/
      base.py                 # Benchmark ABC  <-- the extension seam
      humaneval_plus.py       # first impl (delegates eval to EvalPlus)
    sandbox/                  # docker code-exec runner
    runner/                   # orchestrator: run spec -> results
    results/                  # Postgres store + reporting (views, tables)
  specs/                      # run specs (yaml)
  runs/                       # JSONL export + run manifests (gitignored)
  docs/ARCHITECTURE.md        # this document
```

## 11. Roadmap

**Phase 0 — MVP vertical slice (proves the pipeline).**
HumanEval+ × {one cloud model + one Spark model} → generate via gateway →
sandbox-evaluate via EvalPlus → results table with accuracy + cost/latency.
Exercises every seam: gateway routing (local + cloud), prompt/extract, sandbox,
results+provenance. The Postgres `run`/`sample` schema (D7) and dataset/tool
pinning (D15) apply from this phase, not deferred.

**Phase 1 — Harden.** Endpoint manager (SSH bring-up of Pi + Spark, readiness-
wait + serialized swaps), per-model concurrency tuning, run manifest +
reproducibility, **resumable/idempotent runs (D12)**, reporting tables.

**Phase 2 — Breadth.** Add BigCodeBench (heavier sandbox), then LiveCodeBench
(evaluator + dataset versioning), then Aider polyglot (multi-turn `run()` +
`workspace` sandbox profile).

**Phase 3 — Scale/quality of life.** Dashboards, scheduled runs, model-matrix
sweeps (scheduled grouped by `(box, model)` to amortize vLLM reloads, D13).

## 12. Open questions

- **Cost comparability (D10):** applied with the default (self-hosted labeled,
  never $0, plus a `tok/s` column). Still to confirm: do you want synthetic,
  amortized `$/task` for self-hosted (needs hardware-cost / utilization / $/kWh
  assumptions), and is power metering (smart plug / `tegrastats`) available for
  `Wh/task`?
- **Spark precision & co-location (D13):** applied with the default (one active
  large model per box, swaps serialized). Still to confirm: FP16 vs quantized
  (AWQ/GPTQ) on the Spark, and whether to co-locate multiple small vLLM
  instances.
- **Pi model ceiling:** which Qwen-Coder sizes are worth including given the
  Pi's throughput?

*Resolved since the first draft: gateway host (→ the workstation, §9); results
backend (→ Postgres, D7); dataset/version pinning (→ D15, applies from Phase 0).*
