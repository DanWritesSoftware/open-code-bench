"""Run provenance helpers (D5/D15): git commit + a content hash of the loaded dataset.

Extracted verbatim from scripts/gen_humaneval.py during the Phase-1 framework refactor — same
behaviour, now shared across benchmarks.
"""
from __future__ import annotations

import hashlib
import json
import subprocess


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def dataset_hash(problems: dict) -> str:
    # Hash the FULL problem (prompt + base_input + plus_input + test + entry_point + ...),
    # NOT just the prompt: the "+" in HumanEval+ is the EXTRA test inputs. A prompt-only
    # hash wouldn't change if the evaluator revised plus_input -> false reproducibility.
    blob = json.dumps({k: problems[k] for k in sorted(problems)},
                      sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
