"""Aider polyglot benchmark plugin — the fourth plugin, and the first *agentic* one (§6).

New dimension vs the other three (all single-shot, single-language): **multi-turn editing across
six languages with a test-driven feedback loop** (D14). Each exercise gives the model an
Exercism-style spec + stub file(s); the model must edit the file(s) to pass a hidden test suite.
If the first attempt's tests fail, the test output is fed back and the model gets a second try
(configurable `tries`, default 2 — matching aider's leaderboard protocol). The metric is
`pass_rate_1` (solved on the first try) and `pass_rate_2` (solved within `tries` tries).

Why this breaks the clean generate/score split the other plugins enjoy: the feedback loop *runs
tests between model turns*, so execution is interleaved with generation. This plugin therefore
overrides `run()` and executes each attempt's tests in the sandbox the runner hands it (the runner
builds a sandbox for `multi_turn` benchmarks and passes it into `run()`). By the time generation
finishes, every exercise already has its pass/fail recorded; `evaluate()` is then a pure
aggregation over records.jsonl (no sandbox needed — pass `score <dir> --skip-eval`).

Edit format: **whole-file** (aider's "whole" edit format), not SEARCH/REPLACE — the model returns
the complete updated content of each file it changes, in a fenced block preceded by the file path.
This is deterministic to parse and comparable to the leaderboard's whole-format entries; it is NOT
the default diff format, so numbers are only apples-to-apples with other whole-format runs.

Dataset: the Aider-AI/polyglot-benchmark git repo (Exercism practice exercises for cpp/go/java/
javascript/python/rust). It is not on the HF Hub, so we read a local clone (D15: pin a commit).
Point at it with the `repo_path` option or the POLYGLOT_BENCH_PATH env var; default
`datasets/polyglot-benchmark`. Clone once:
    git clone https://github.com/Aider-AI/polyglot-benchmark datasets/polyglot-benchmark

Scoring image: ocb-polyglot-exec (docker/exec-polyglot) — a fat, offline image carrying all six
toolchains + pre-warmed dependency caches so tests run under --network=none (D11).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from ocb.benchmarks.base import Benchmark, Message, Metrics, Solution, Task

# Languages the polyglot repo ships (top-level dir name -> language key). Also the score order.
LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")

SYSTEM_PROMPT = (
    "You are an expert software engineer fluent in C++, Go, Java, JavaScript, Python and Rust. "
    "You are given a coding exercise and the current contents of the file(s) you may edit. Modify "
    "the code so that it satisfies the specification and passes the (hidden) test suite."
)

# Whole-file edit-format instructions, appended to the first user turn. Mirrors aider's "whole"
# format so the parser can reliably recover each file's new contents.
EDIT_FORMAT_INSTRUCTIONS = (
    "\n\nTo make changes, output the ENTIRE, updated contents of each file you want to change. "
    "Format every file as its path on a line by itself, immediately followed by a fenced code "
    "block containing the file's full new content, like this:\n\n"
    "path/to/file.ext\n"
    "```\n"
    "<the complete file content>\n"
    "```\n\n"
    "Rules:\n"
    "- Output the WHOLE file, not a diff or a fragment.\n"
    "- Only edit the file(s) listed above; do not create new files or edit the tests.\n"
    "- Keep the same file path and public interface the tests expect."
)


def _normalize_path_line(s: str) -> str:
    """Strip the decorations models wrap a filename line in (backticks, headers, 'File:', bullets,
    bold/italic markers), from both ends, leaving a bare path."""
    s = re.sub(r"^\s*(#{1,6}\s+|[-*>]\s+)", "", s.strip())   # header / bullet / quote marker
    for pre in ("File:", "file:", "FILE:", "Filename:", "filename:", "Path:", "path:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
    s = s.strip().strip("`").strip("*").strip("`").strip().strip(":").strip()
    return s


def _parse_whole_file_edits(completion: str, solution_files: list[str]) -> dict[str, str]:
    """Recover {relpath: new_content} from a whole-file-format completion.

    For each fenced block, the file it targets is the nearest matching path line in the 3 non-empty
    lines above the opening fence; if none matches and there is exactly one editable file, the block
    is assigned to it (the common single-file case). The last block wins for a given file."""
    by_base: dict[str, str] = {}
    for rp in solution_files:
        by_base.setdefault(Path(rp).name, rp)

    def match_path(line: str) -> str | None:
        s = _normalize_path_line(line)
        if not s:
            return None
        s = s.replace("\\", "/")
        for rp in solution_files:
            if s == rp or s.endswith("/" + rp) or rp.endswith("/" + s):
                return rp
        return by_base.get(Path(s).name)

    lines = completion.split("\n")
    n = len(lines)
    edits: dict[str, str] = {}
    i = 0
    while i < n:
        if lines[i].lstrip().startswith("```"):
            j = i + 1
            while j < n and not lines[j].lstrip().startswith("```"):
                j += 1
            content = "\n".join(lines[i + 1:j])
            fname, k, seen = None, i - 1, 0
            while k >= 0 and seen < 3:
                if lines[k].strip():
                    fname = match_path(lines[k])
                    if fname:
                        break
                    seen += 1
                k -= 1
            if fname is None and len(solution_files) == 1:
                fname = solution_files[0]
            if fname is not None:
                edits[fname] = content
            i = j + 1
        else:
            i += 1
    return edits


def _tail(text: str, n: int = 4000) -> str:
    """Keep the last n chars of test output for feedback (test suites can be very chatty)."""
    return text if len(text) <= n else "...(truncated)...\n" + text[-n:]


class AiderPolyglot(Benchmark):
    name = "aider_polyglot"
    conversation_mode = "multi_turn"
    system_prompt = SYSTEM_PROMPT
    sandbox_image = "ocb-polyglot-exec:0.1.0"   # fat offline image, all 6 toolchains (docker/exec-polyglot)
    run_prefix = "polyglot"
    sandbox_read_only = False    # compilers/build tools write artifacts (target/, build/, node_modules)
    sandbox_auto_confirm = False

    # Test command run inside /work (the mounted exercise dir), per language. cpp/javascript use
    # helper scripts baked into the image (cmake build+ctest; npm offline install+jest). Java/Rust
    # run offline against pre-warmed caches. {tests} is replaced by the exercise's test file paths.
    TEST_COMMANDS = {
        "python": "python -m pytest -q {tests}",
        "go": "go test ./...",
        "rust": "cargo test --offline -- --include-ignored",
        "java": "./gradlew test --offline --no-daemon --console=plain",
        "javascript": "/opt/ocb/js-test.sh",
        "cpp": "/opt/ocb/cpp-test.sh",
    }

    def __init__(self, repo_path: str | None = None, languages: list[str] | None = None,
                 tries: int = 2, commit: str | None = None, per_test_timeout: int = 300):
        self.repo_path = Path(repo_path or os.environ.get("POLYGLOT_BENCH_PATH",
                                                          "datasets/polyglot-benchmark"))
        self.languages = tuple(languages) if languages else LANGUAGES
        for lg in self.languages:
            if lg not in self.TEST_COMMANDS:
                raise ValueError(f"unknown language {lg!r}; known: {sorted(self.TEST_COMMANDS)}")
        self.tries = int(tries)
        if self.tries < 1:
            raise ValueError("tries must be >= 1")
        self.per_test_timeout = int(per_test_timeout)
        self.commit = commit            # pinned dataset commit for provenance (D15), if known
        self._tasks: list[Task] | None = None
        langs = "-".join(self.languages) if len(self.languages) < len(LANGUAGES) else "all"
        self.version = f"aider-polyglot-{langs}-try{self.tries}"

    # ---- dataset (read the local git clone; Exercism practice layout) ----
    def _practice_dir(self, lang: str) -> Path:
        # The repo groups exercises under <lang>/exercises/practice/<exercise>/.
        return self.repo_path / lang / "exercises" / "practice"

    def _load(self) -> list[Task]:
        if self._tasks is not None:
            return self._tasks
        if not self.repo_path.is_dir():
            raise FileNotFoundError(
                f"polyglot-benchmark clone not found at {self.repo_path} — set repo_path or "
                "POLYGLOT_BENCH_PATH, or clone: git clone "
                "https://github.com/Aider-AI/polyglot-benchmark <path>")
        tasks: list[Task] = []
        for lang in self.languages:
            pdir = self._practice_dir(lang)
            if not pdir.is_dir():
                continue
            for ex_dir in sorted(p for p in pdir.iterdir() if p.is_dir()):
                cfg_path = ex_dir / ".meta" / "config.json"
                instr_path = ex_dir / ".docs" / "instructions.md"
                if not cfg_path.is_file() or not instr_path.is_file():
                    continue
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                files = cfg.get("files", {})
                solution_files = list(files.get("solution", []))
                test_files = list(files.get("test", []))
                if not solution_files or not test_files:
                    continue
                task = Task(
                    task_id=f"{lang}/{ex_dir.name}",
                    data={
                        "language": lang,
                        "exercise": ex_dir.name,
                        "dir": str(ex_dir),
                        "instructions": self._read_instructions(ex_dir),
                        "solution_files": solution_files,
                        "test_files": test_files,
                        "stub_contents": {rp: self._read_file(ex_dir / rp) for rp in solution_files},
                    },
                )
                tasks.append(task)
        self._tasks = tasks
        return tasks

    @staticmethod
    def _read_file(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return ""

    def _read_instructions(self, ex_dir: Path) -> str:
        docs = ex_dir / ".docs"
        parts = []
        for fn in ("introduction.md", "instructions.md", "instructions.append.md"):
            f = docs / fn
            if f.is_file():
                parts.append(self._read_file(f).strip())
        return "\n\n".join(p for p in parts if p)

    def load_dataset(self, limit: int = 0) -> list[Task]:
        tasks = self._load()
        return tasks[:limit] if limit else tasks

    # ---- prompting ----
    def build_prompt(self, task: Task) -> list[Message]:
        """First-turn prompt: spec + current file contents + whole-file format instructions."""
        d = task.data
        files_block = "\n\n".join(
            f"{rp}\n```\n{d['stub_contents'].get(rp, '')}\n```" for rp in d["solution_files"])
        user = (
            f"# Exercise: {d['exercise']} ({d['language']})\n\n"
            f"{d['instructions']}\n\n"
            f"## Files you may edit\n\n{files_block}\n"
            f"{EDIT_FORMAT_INSTRUCTIONS}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _feedback_prompt(self, task: Task, test_output: str) -> str:
        return (
            "The tests did not pass. Here is the test output:\n\n"
            f"```\n{_tail(test_output)}\n```\n\n"
            "Analyse the failure and output the full, corrected contents of the file(s) again, "
            "using the same whole-file format (path line, then a fenced block per file)."
        )

    # ---- agentic multi-turn driver (overrides the single-shot default, D14) ----
    def run(self, task: Task, client, *, model: str, sampling: dict, run_id: str,
            sample_index: int = 0, sandbox=None) -> Solution:
        if sandbox is None:
            raise ValueError("AiderPolyglot.run() needs a SandboxRunner (multi_turn benchmark)")
        d = task.data
        messages = self.build_prompt(task)
        attempts: list[dict] = []
        last_text = ""
        prompt_tok = comp_tok = 0
        t0 = time.time()
        passed = False
        passed_on = None

        with tempfile.TemporaryDirectory(prefix="ocb-polyglot-") as tmp:
            work = Path(tmp) / task.data["exercise"]
            shutil.copytree(d["dir"], work, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".git"))

            for attempt in range(1, self.tries + 1):
                rec: dict = {"attempt": attempt}
                try:
                    comp = client.complete(
                        messages, model=model, temperature=sampling["temperature"],
                        max_tokens=sampling["max_tokens"], run_id=run_id, benchmark=self.name,
                        task_id=task.task_id, sample_index=sample_index,
                        num_ctx=sampling.get("num_ctx"))
                except Exception as e:                       # generation infra failure (D12)
                    rec.update({"gen_status": "infra_error", "error": repr(e)})
                    attempts.append(rec)
                    break
                last_text = comp.content or ""
                prompt_tok += comp.prompt_tokens or 0
                comp_tok += comp.completion_tokens or 0
                gen_status = "truncated" if comp.finish_reason == "length" else "ok"
                rec.update({"gen_status": gen_status, "finish_reason": comp.finish_reason,
                            "completion_tokens": comp.completion_tokens, "latency_s": comp.latency_s})

                edits = _parse_whole_file_edits(last_text, d["solution_files"])
                rec["edited_files"] = sorted(edits)
                # Apply edits onto the working copy; restore original test files so the model can't
                # (accidentally or otherwise) weaken the tests.
                for rp, content in edits.items():
                    (work / rp).parent.mkdir(parents=True, exist_ok=True)
                    (work / rp).write_text(content, encoding="utf-8")
                for rp in d["test_files"]:
                    src = Path(d["dir"]) / rp
                    if src.is_file():
                        (work / rp).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, work / rp)

                if not edits:
                    rec.update({"test_exit": None, "test_passed": False,
                                "note": "no parseable file edits in completion"})
                    attempts.append(rec)
                    messages += [{"role": "assistant", "content": last_text},
                                 {"role": "user", "content": self._feedback_prompt(
                                     task, "No file edits were found in your reply. Output the full "
                                     "file using the required path-then-fenced-block format.")}]
                    continue

                try:
                    exit_code, output = self._run_tests(task, work, sandbox)
                except Exception as e:                       # sandbox/ssh infra failure
                    rec.update({"test_exit": None, "test_passed": None,
                                "sandbox_error": repr(e)})
                    attempts.append(rec)
                    break
                test_passed = exit_code == 0
                rec.update({"test_exit": exit_code, "test_passed": test_passed,
                            "test_output_tail": _tail(output, 1500)})
                attempts.append(rec)
                if test_passed:
                    passed, passed_on = True, attempt
                    break
                if attempt < self.tries:
                    messages += [{"role": "assistant", "content": last_text},
                                 {"role": "user", "content": self._feedback_prompt(task, output)}]

        # A record is "fairly attempted" (D12) if at least one turn produced a completion...
        produced = any(a.get("gen_status") in ("ok", "truncated") for a in attempts)
        overall = "ok" if produced else "infra_error"
        # ...and "evaluable" only if at least one attempt yielded a real test verdict. If every
        # attempt's tests died on a sandbox/ssh error (test_passed None), we can't fairly call the
        # exercise failed — evaluate() excludes it from the pass-rate denominator (D12), rather than
        # scoring an infra problem as a wrong answer. (A no-edit turn is test_passed=False — a
        # genuine failure — so it stays evaluable.)
        evaluable = any(a.get("test_passed") in (True, False) for a in attempts)
        extra = {
            "language": d["language"], "exercise": d["exercise"], "tries": self.tries,
            "attempts_used": len(attempts), "passed": passed, "passed_on_attempt": passed_on,
            "evaluable": evaluable, "attempts": attempts,
        }
        return Solution(
            task_id=task.task_id, sample_index=sample_index, text=last_text, gen_status=overall,
            finish_reason=attempts[-1].get("finish_reason") if attempts else None,
            prompt_tokens=prompt_tok or None, completion_tokens=comp_tok or None,
            latency_s=round(time.time() - t0, 2),
            error=attempts[-1].get("error") if attempts and not produced else None,
            extra=extra)

    def _run_tests(self, task: Task, work: Path, sandbox) -> tuple[int, str]:
        lang = task.data["language"]
        cmd = self.TEST_COMMANDS[lang].format(
            tests=" ".join(task.data["test_files"]))
        # Bound each test run inside the container (coreutils `timeout`) as well as at the
        # subprocess level; a nonzero exit is a failing test, not an infra error.
        script = f"cd /work && timeout {self.per_test_timeout}s {cmd}"
        inner = ["bash", "-lc", script]
        return sandbox.run_dir_capture(work, inner, timeout=self.per_test_timeout + 60)

    # ---- provenance ----
    def dataset_hash(self) -> str:
        tasks = self._load()
        blob = json.dumps(
            [{"task_id": t.task_id, "instructions": t.data["instructions"],
              "solution_files": t.data["solution_files"],
              "stub_contents": t.data["stub_contents"]} for t in tasks],
            sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def backend_info(self, model: str) -> dict:
        if model.startswith("qwen-pi"):
            return {"server": "ollama", "model_logical": model}
        return {"server": "vllm", "served_model_logical": model,
                "note": "served via vLLM; dtype/quant set at vLLM launch"}

    # ---- scoring seam: pure aggregation (tests already ran during generation) ----
    def evaluate(self, run_dir, *, sandbox=None, skip_eval: bool = False, **_) -> Metrics:
        """AiderPolyglot runs its tests during generation (the feedback loop needs them), so scoring
        is a pure aggregation over records.jsonl — no sandbox required. Run `score <dir> --skip-eval`."""
        run_dir = Path(run_dir)
        records = [json.loads(l) for l in
                   (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

        per_lang: dict[str, dict[str, int]] = {}
        n_fair = n_infra = n_sandbox_err = n_pass1 = n_pass = 0
        rows = []
        for rec in records:
            gs = rec.get("gen_status")
            lang = rec.get("language") or (str(rec.get("task_id", "")).split("/", 1) + [""])[0]
            passed = bool(rec.get("passed"))
            passed_on = rec.get("passed_on_attempt")
            lang_stats = per_lang.setdefault(
                lang, {"fair": 0, "pass1": 0, "pass": 0, "infra": 0, "sandbox_err": 0})
            if gs == "infra_error":
                n_infra += 1
                lang_stats["infra"] += 1
                rows.append({**rec, "eval_status": None}); continue
            if not rec.get("evaluable", True):     # tests never returned a verdict (sandbox/ssh) — exclude
                n_sandbox_err += 1
                lang_stats["sandbox_err"] += 1
                rows.append({**rec, "eval_status": "sandbox_error"}); continue
            n_fair += 1
            lang_stats["fair"] += 1
            if passed:
                n_pass += 1
                lang_stats["pass"] += 1
                if passed_on == 1:
                    n_pass1 += 1
                    lang_stats["pass1"] += 1
            rows.append({**rec, "eval_status": "passed" if passed else "failed"})

        (run_dir / "scores.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        lang_rates = {lg: {
            "fairly_attempted": s["fair"], "infra_error": s["infra"], "sandbox_error": s["sandbox_err"],
            "pass_rate_1": (s["pass1"] / s["fair"]) if s["fair"] else None,
            f"pass_rate_{self.tries}": (s["pass"] / s["fair"]) if s["fair"] else None,
        } for lg, s in sorted(per_lang.items())}
        summary = {
            "run_id": run_dir.name, "benchmark": self.name, "version": self.version,
            "tries": self.tries, "total_records": len(records),
            "fairly_attempted": n_fair, "gen_infra_error": n_infra, "sandbox_error": n_sandbox_err,
            "passed_try1": n_pass1, "passed_withintries": n_pass,
            "pass_rate_1": (n_pass1 / n_fair) if n_fair else None,
            f"pass_rate_{self.tries}": (n_pass / n_fair) if n_fair else None,
            "per_language": lang_rates,
            "completeness": (f"{n_fair}/{len(records)} fairly attempted, "
                             f"{n_infra} infra, {n_sandbox_err} sandbox_error"),
        }
        (run_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return Metrics(summary=summary)
