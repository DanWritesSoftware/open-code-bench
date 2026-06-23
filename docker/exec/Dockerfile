# open-code-bench — hardened "exec" sandbox for HumanEval+ scoring (D11).
#
# WHY THIS EXISTS: model-generated code is untrusted. It executes ONLY inside this
# container, never on the gateway host (D6/D9 §9). EvalPlus's scorer runs *inside*
# here too (D1/D11) — we delegate scoring but never treat a tool's built-in exec as
# the security boundary; this container is the boundary.
#
# The container is run with the D11 hardening flags by scripts/score_humaneval.py
# (NOT baked here, so they stay auditable at the call site):
#   --network=none --cap-drop=ALL --read-only --pids-limit=256
#   --cpus=<n> --memory=<m> --user (the baked non-root user) + a per-run timeout,
#   with a writable bind-mount at /work for samples in + results out, /tmp tmpfs scratch.
#
# DATASET IS PRE-BAKED AT BUILD TIME. EvalPlus would otherwise fetch HumanEval+ from
# the network on first use — impossible under --network=none. We warm the cache during
# build (network available) so the runtime container is fully offline.
#
# Build (on the sandbox host, e.g. the T14):
#   docker build -t ocb-exec:0.3.1 docker/exec
#
# Pin (D15): EVALPLUS_VERSION matches requirements.txt; the HumanEval+ dataset version
# is whatever evalplus 0.3.1 pins (HumanEvalPlus v0.1.10), so tool + dataset stay locked.
FROM python:3.12-slim

ARG EVALPLUS_VERSION=0.3.1

# Non-root from the start (D11). uid 10001 is unprivileged and not a real host user.
RUN useradd --create-home --uid 10001 sandbox

RUN pip install --no-cache-dir "evalplus==${EVALPLUS_VERSION}"

USER sandbox
ENV HOME=/home/sandbox
WORKDIR /home/sandbox

# Warm BOTH caches as the sandbox user while the network is up and the rootfs is writable:
#   1. the dataset jsonl  (get_human_eval_plus decompresses it into ~/.cache/evalplus)
#   2. the ground-truth pickle  (evalplus.evaluate writes ~/.cache/evalplus/<hash>.pkl on
#      first run by executing the canonical solutions — trusted code, fine at build time).
# Pre-baking #2 is REQUIRED: at runtime the rootfs is --read-only, so that pickle write
# would fail. With both cached, the runtime container reads only and needs no network.
# World-readable (a+rX) so the cache still resolves if run with --user <host-uid>.
RUN python -c "from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash; \
from evalplus.evaluate import get_groundtruth; \
p = get_human_eval_plus(); h = get_human_eval_plus_hash(); get_groundtruth(p, h, [])" \
 && chmod -R a+rX /home/sandbox/.cache

# Runtime work dir: samples in, eval_results.json out. The runner bind-mounts a writable
# host dir here; the rest of the rootfs is mounted read-only at run time.
WORKDIR /work

# No ENTRYPOINT on purpose: the runner passes the exact evalplus.evaluate command so the
# scoring invocation is explicit in logs, e.g.:
#   python -m evalplus.evaluate humaneval --samples /work/samples-sanitized.jsonl
