"""Phase-0 smoke test: verify runner-sent params reach the Pi through the gateway.

Sends a HumanEval-style completion request TWICE with temperature=0 and an explicit
system prompt (so Ollama doesn't inject Qwen's default one), then checks:
  * finish_reason  -> expect 'stop', not 'length' (no truncation at max_tokens)
  * determinism    -> identical content across both calls proves temp=0 / greedy
                      actually took effect, despite drop_params silently dropping
                      unsupported params.

Requires the gateway running on localhost:4000.
Run:  .venv\\Scripts\\python.exe scripts\\smoke_gateway.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-noauth")  # no master_key set

PROMPT = '''Complete this Python function. Respond with code only.

from typing import List


def below_zero(operations: List[int]) -> bool:
    """Given deposits/withdrawals on an account starting at zero balance, return True if
    the balance ever falls below zero, otherwise False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """'''

req = dict(
    model="qwen-pi-1.5b",
    temperature=0,            # pass@1 baseline; Ollama's default is 0.8
    max_tokens=512,           # generous so a real solution isn't truncated
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant. Respond with Python code only."},
        {"role": "user", "content": PROMPT},
    ],
)


def call(n):
    r = client.chat.completions.create(**req)
    ch = r.choices[0]
    print(f"--- call {n}: finish_reason={ch.finish_reason}, "
          f"completion_tokens={r.usage.completion_tokens} ---")
    return ch.message.content


c1 = call(1)
c2 = call(2)
print("\n=== completion (call 1) ===")
print(c1)
print("\n=== temp=0 verification (determinism) ===")
print("IDENTICAL  ->  greedy/temp=0 took effect" if c1 == c2
      else "DIFFERENT  ->  greedy NOT active; sampling param did not take effect")
