# open-code-bench — hardened, OFFLINE LiveCodeBench exec image (D11).
#
# Runs lcb_runner's codegeneration evaluator on model outputs. Unlike the BigCodeBench image (which
# extends an official evaluator base), LiveCodeBench ships no eval-only image, so we build one:
#   * clone lcb_runner at a PINNED commit (D15);
#   * install ONLY the scoring deps (datasets<4.0 to run the trust_remote_code builder + numpy/
#     pandas/pebble/tqdm/anthropic) — NOT the generation stack (torch/vLLM/vendor SDKs). Our thin
#     wrapper `ocb_lcb_eval.py` is the ENTRYPOINT and sidesteps lcb_runner.runner.parser's torch;
#   * BAKE the release_v5 dataset files into the image (network NOT needed at build) then force HF
#     offline, so the runtime container scores with --network=none.
#
# Populate the build context before building (one-time, or when re-pinning the dataset):
#   mkdir -p docker/exec-lcb/lcb-data
#   # copy code_generation_lite.py, README.md, test.jsonl..test6.jsonl from a HF cache snapshot
#   # (~/.cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/<rev>/)
#   # or download fresh via huggingface_hub.hf_hub_download for each file.
#   # NOTE: the builder script references all six test*.jsonl even when scoring only release_v5 —
#   # all six must be present or the offline load_dataset call below fails.
#
# Build (on the sandbox host, amd64):
#   docker build -t ocb-lcb-exec:0.1.0 docker/exec-lcb
#
# Why baked-in instead of downloaded at build time: this used to `load_dataset(...,
# trust_remote_code=True)` straight from the Hub during the build, but the Hub's unauthenticated
# rate limit can stall that download indefinitely on a slow/throttled host — baking in exact,
# pinned bytes is also strictly better for D15 dataset-pinning reproducibility regardless.
#
# The evaluator + datasets write lock/temp files at runtime, so the runner uses the looser writable-
# rootfs profile for this benchmark (sandbox_read_only=False); --network=none is still enforced.
FROM python:3.11-slim

# datasets<4.0 still runs the code_generation_lite trust_remote_code builder that datasets>=4.0
# refuses; the rest are what lcb_runner's scoring path imports (pandas is imported into the
# candidate-exec globals; anthropic is an unguarded import in lcb_runner/prompts/self_repair.py).
RUN pip install --no-cache-dir \
        "datasets<4.0" \
        numpy \
        pandas \
        pebble \
        tqdm \
        attrs \
        anthropic

# lcb_runner at the pinned commit (shallow-ish clone, then checkout). git is needed only here.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone https://github.com/LiveCodeBench/LiveCodeBench.git /opt/lcb \
    && cd /opt/lcb && git checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/opt/lcb

# Bake the LiveCodeBench dataset files locally (trust_remote_code builder script + test*.jsonl).
# The dir MUST be named 'code_generation_lite' to match the builder script's basename — otherwise
# `datasets` silently ignores the .py and falls back to its generic JSON loader (which then rejects
# the version_tag kwarg the builder config needs).
COPY lcb-data /opt/code_generation_lite
ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# Pre-build the arrow cache offline from the local dir (same path/config the runtime wrapper uses,
# so runtime reuses it via memory-mapping). No network. Force a tiny writer_batch_size: LCB rows
# carry huge base64 private_test_cases, and the datasets-default ~1000-row buffer can OOM-kill a
# memory-constrained build host — small batches keep peak memory low.
RUN python -c "from datasets import load_dataset_builder; b=load_dataset_builder('/opt/code_generation_lite', version_tag='release_v5', trust_remote_code=True); b._writer_batch_size=8; b.download_and_prepare()"

COPY ocb_lcb_eval.py /opt/ocb_lcb_eval.py

# Non-root (D11). The dataset cache lives in this user's $HOME (warmed above as root, so copy it in).
RUN useradd -m lcb && cp -r /root/.cache /home/lcb/.cache && chown -R lcb:lcb /home/lcb/.cache
USER lcb
WORKDIR /work
ENTRYPOINT ["python", "/opt/ocb_lcb_eval.py"]
