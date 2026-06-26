"""Benchmark registry — name -> Benchmark class (ARCHITECTURE.md §6).

Add a benchmark = implement the Benchmark interface and register it here.
"""
from __future__ import annotations

from ocb.benchmarks.base import Benchmark
from ocb.benchmarks.humaneval_plus import HumanEvalPlus

_REGISTRY: dict[str, type[Benchmark]] = {
    HumanEvalPlus.name: HumanEvalPlus,
}


def get_benchmark(name: str) -> Benchmark:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(_REGISTRY)}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)
