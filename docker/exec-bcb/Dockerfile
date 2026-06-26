# open-code-bench — hardened, OFFLINE BigCodeBench exec image (D11).
#
# Extends the official evaluator image (the ~139 task libraries + the `bigcodebench.evaluate`
# ENTRYPOINT) by PRE-BAKING the dataset, so the runtime container runs with --network=none
# --read-only. The stock image downloads bigcode/bigcodebench[-hard] from the HF Hub at eval
# time, which fails under --network=none (ConnectionError). We warm both subsets at build time
# (network available) and then force HF offline so the runtime never touches the Hub.
#
# Build (on the sandbox host / Mint, amd64):
#   docker build -t ocb-bcb-exec:0.2.4 docker/exec-bcb
#
# Pin (D15): base tag v0.2.4 == the dataset/tool version baked in; dataset version v0.1.4.
FROM bigcodebench/bigcodebench-evaluate:v0.2.4

# Warm the dataset cache for both subsets while the network is up. The base image runs as a
# non-root user, so the cache lands in that user's $HOME/.cache; at runtime the same user reads
# it (no chmod needed), and it's baked into the layer so --read-only just reads it.
RUN python -c "from bigcodebench.data import get_bigcodebench; get_bigcodebench(subset='hard'); get_bigcodebench(subset='full')"

# Pre-compute + cache the ground-truth pickle (expected outputs) for the hard subset, so the runtime
# LOADS it instead of WRITING it (blocked by --read-only). ~4 min; runs the canonical solutions
# (trusted; their network I/O is mocked, so it works offline). --check_gt_only bypasses samples.
RUN bigcodebench.evaluate --check_gt_only --split complete --subset hard || true

# Runtime: never reach the Hub — use the baked cache only.
ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Inherit the base ENTRYPOINT (`bigcodebench.evaluate`); the runner passes its args.
