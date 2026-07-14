"""Thin offline wrapper over lcb_runner's codegeneration evaluator (ocb-lcb-exec ENTRYPOINT).

Replicates lcb_runner/runner/custom_evaluator.py's codegeneration path, but builds `args` as a
plain namespace instead of going through lcb_runner.runner.parser — the parser does `import torch`
and drags in the whole generation stack (vLLM, vendor SDKs). Scoring needs none of it, so this
wrapper keeps the image lean (datasets + numpy + pandas + pebble + anthropic).

Input : a custom_output_file — a JSON list of {"question_id", "code_list": [code]} (already
        fence-extracted by the ocb LiveCodeBench plugin).
Output : <stem>_codegeneration_output_eval_all.json — one dict per question with graded_list + pass@1.

Offline dataset: the image bakes the LiveCodeBench release files at /opt/code_generation_lite (dir
name MUST match the trust_remote_code builder script's basename, or `datasets` silently falls back
to its generic JSON loader and rejects the `version_tag` kwarg). lcb_runner loads the dataset via
load_dataset("livecodebench/code_generation_lite", trust_remote_code=True), which needs the Hub;
`_use_local_dataset()` redirects that repo id to the baked local dir so both the image build and
runtime scoring work with no Hub access (--network=none at runtime).

cwd: lcb_runner's prompt module opens a path relative to the repo root ("lcb_runner/prompts/
few_shot_examples/...") at IMPORT time, so this process must chdir to /opt/lcb (the clone root)
before importing lcb_runner — the sandbox runner mounts the run dir at /work and everything this
wrapper touches (custom_output_file etc.) is passed in as an absolute path, so the chdir is safe.

Memory: lcb_runner's load_code_generation_dataset() unconditionally constructs a CodeGenerationProblem
for EVERY row in the release (each construction decompresses base64+zlib+pickle private_test_cases —
these can be large) and only filters by start_date/end_date AFTER all of them are built. That means a
--start-date/--end-date window does nothing to reduce peak memory with the stock loader — scoring the
full release_v5 (880 problems) OOM-killed a memory-constrained (11GB) sandbox host even when only a
368-problem post-cutoff window was requested. `_use_frugal_dataset_loader()` replaces the loader with
one that filters the raw (undecompressed) rows by contest_date BEFORE construction, so a date-windowed
score only ever decompresses the problems actually in that window.
"""
import argparse
import json
import os
from datetime import datetime
from types import SimpleNamespace

os.chdir("/opt/lcb")

from lcb_runner.utils.scenarios import Scenario
import lcb_runner.runner.scenario_router as scenario_router
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    sort_and_extract_save_results,
    get_metrics,
)
from lcb_runner.evaluation import extract_instance_results

LOCAL_DATASET_DIR = "/opt/code_generation_lite"


def _use_local_dataset() -> None:
    """Redirect load_dataset('livecodebench/code_generation_lite', ...) to the image-baked local
    snapshot dir. Rebinds the symbol in `datasets` and in every module that already imported it,
    so it works regardless of lcb_runner's import style."""
    import sys
    import datasets

    orig = datasets.load_dataset
    if getattr(orig, "_ocb_patched", False):
        return

    def patched(path, *args, **kwargs):
        if isinstance(path, str) and path.endswith("code_generation_lite"):
            path = LOCAL_DATASET_DIR
        return orig(path, *args, **kwargs)

    patched._ocb_patched = True
    datasets.load_dataset = patched
    for module in list(sys.modules.values()):
        try:
            if getattr(module, "load_dataset", None) is orig:
                module.load_dataset = patched
        except Exception:
            pass


def _use_frugal_dataset_loader() -> None:
    """Replace scenario_router's load_code_generation_dataset with one that filters raw rows by
    contest_date BEFORE constructing CodeGenerationProblem (see module docstring: construction is
    what decompresses private_test_cases and is the actual memory cost, not row count post-filter).
    scenario_router imported the name directly (`from lcb_runner.benchmarks import
    load_code_generation_dataset`), so it must be rebound on the scenario_router module object
    itself — patching the source module's attribute after that import already happened has no effect.

    IMPORTANT: even reading `contest_date` per row via plain Python iteration (`list(dataset)` or
    `dataset.filter(fn)` with a row-wise predicate) forces `datasets` to materialize EVERY column for
    EVERY row, including the huge private/public_test_cases strings — that materialization alone (not
    the later CodeGenerationProblem decompression) is what spiked memory to ~7.7GB for all 880 rows
    even when only computing which rows to keep. Fix: `.select_columns()` down to just the columns the
    date filter needs, compute the surviving row indices from THAT lightweight view, then `.select()`
    only those indices on the full dataset — the heavy columns are only ever touched for rows that
    survive the filter."""
    from datasets import load_dataset
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    def frugal_load(release_version="release_v1", start_date=None, end_date=None):
        dataset = load_dataset(LOCAL_DATASET_DIR, split="test",
                                version_tag=release_version, trust_remote_code=True)
        n_total = len(dataset)
        if start_date is not None or end_date is not None:
            light = dataset.select_columns(["question_id", "contest_date"])
            keep_idx = range(len(light))
            if start_date is not None:
                d0 = datetime.strptime(start_date, "%Y-%m-%d")
                keep_idx = [i for i in keep_idx if d0 <= datetime.fromisoformat(light[i]["contest_date"])]
            if end_date is not None:
                d1 = datetime.strptime(end_date, "%Y-%m-%d")
                keep_idx = [i for i in keep_idx if datetime.fromisoformat(light[i]["contest_date"]) <= d1]
            dataset = dataset.select(keep_idx)
        result = [CodeGenerationProblem(**r) for r in dataset]
        print(f"[ocb-lcb-eval] frugal loader: constructed {len(result)}/{n_total} problems "
              f"(date-filtered before decompression)", flush=True)
        return result

    scenario_router.load_code_generation_dataset = frugal_load


def main() -> None:
    _use_local_dataset()
    _use_frugal_dataset_loader()

    ap = argparse.ArgumentParser()
    ap.add_argument("--custom_output_file", required=True)
    ap.add_argument("--release_version", required=True)
    ap.add_argument("--start_date", default=None)
    ap.add_argument("--end_date", default=None)
    ap.add_argument("--num_process_evaluate", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=6)
    a = ap.parse_args()

    args = SimpleNamespace(
        scenario=Scenario.codegeneration, release_version=a.release_version,
        not_fast=False, start_date=a.start_date, end_date=a.end_date,
        num_process_evaluate=a.num_process_evaluate, timeout=a.timeout,
        custom_output_file=a.custom_output_file, custom_output_save_name=None,
    )

    benchmark, _ = build_prompt_benchmark(args)

    with open(args.custom_output_file, "r", encoding="utf-8") as f:
        custom_outputs = json.load(f)
    assert isinstance(custom_outputs, list)
    assert len(custom_outputs) == len(benchmark), f"{len(custom_outputs)} != {len(benchmark)}"
    # dict form -> code_list, aligned to the benchmark's question_id sort (custom_evaluator's logic)
    custom_outputs = [c["code_list"] for c in
                      sorted(custom_outputs, key=lambda x: str(x["question_id"]))]

    save_results = [inst.insert_output(o, o) for inst, o in zip(benchmark, custom_outputs)]
    save_results, combined = sort_and_extract_save_results(args.scenario, save_results)

    metrics = get_metrics(args.scenario, args, benchmark, combined)
    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval = [
        inst.insert_output_evaluation(o, e, g, metadata=m)
        for inst, (o, e), g, m in zip(benchmark, combined, graded, metadatas)
    ]

    out = args.custom_output_file[:-5] + "_codegeneration_output_eval_all.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(save_eval, f)
    print(f"[ocb-lcb-eval] wrote {out}  (aggregate pass@1={metrics[0].get('pass@1')})")


if __name__ == "__main__":
    main()
